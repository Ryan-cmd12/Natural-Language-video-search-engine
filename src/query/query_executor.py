from src.query.result_models import (
    CandidateWindow,
)

from src.storage.track_store import (
    TrackStore,
)

from src.query.entity_resolver import (
    EntityResolver,
)

from src.query.attribute_query import (
    AttributeFilter,
    AttributeFilterResult,
)

from src.query.spatial_relationship_filter import (
    SpatialRelationshipFilter,
)

from src.query.window_frame_sampler import (
    WindowFrameSampler,
)

import tempfile

from src.query.temporal_filter import (
    TemporalFilter,
)

class QueryExecutor:

    def __init__(
        self,
        track_store: TrackStore,
        attribute_verifier=None,
        video_verifier=None,
        window_frame_sampler=None,
    ):
        from src.query.temporal_filter import (
            TemporalFilter,
        )

        self.video_verifier = (
            video_verifier
        )

        self.window_frame_sampler = (
            window_frame_sampler
            or
            WindowFrameSampler(
                sample_count=8
            )
        )
        self.spatial_filter = (
            SpatialRelationshipFilter()
        )

        self.track_store = (
            track_store
        )

        self.entity_resolver = (
            EntityResolver(
                track_store=
                    track_store
            )
        )

        self.attribute_filter = (
            AttributeFilter()
        )

        self.attribute_verifier = (
            attribute_verifier
        )
    # ==================================================
    # MAIN EXECUTION
    # ==================================================

    def execute(
        self,
        plan,
        video_id: str,
    ):

        step_outputs = {}

        for step in plan.steps:

            operation = (
                step.operation
            )

            print(
                f"\n[{step.step_id}] "
                f"{operation}"
            )

            # ------------------------------------------
            # TRACK LOOKUP
            # ------------------------------------------

            if operation == "TRACK_LOOKUP":

                result = (
                    self._execute_track_lookup(
                        step,
                        video_id,
                    )
                )

            #-------------------------------------------
            # Attribute filter
            #-------------------------------------------
            elif operation == "ATTRIBUTE_FILTER":

                result = (
                    self._execute_attribute_filter(
                        step,
                        step_outputs,
                    )
                )

            # ------------------------------------------
            # TEMPORAL OVERLAP
            # ------------------------------------------

            elif operation == "TEMPORAL_OVERLAP":

                result = (
                    self._execute_temporal_overlap(
                        step,
                        step_outputs,
                    )
                )

            # ------------------------------------------
            # BUILD WINDOWS
            # ------------------------------------------

            elif (
                operation
                == "BUILD_CANDIDATE_WINDOWS"
            ):

                result = (
                    self._execute_build_candidates(
                        step,
                        step_outputs,
                    )
                )

            # ------------------------------------------
            # RANKING
            # ------------------------------------------

            elif operation == "RANK_RESULTS":

                result = (
                    self._execute_rank_results(
                        step,
                        step_outputs,
                    )
                )

            #-------------------------------------------
            # Attribute verify
            #-------------------------------------------
            elif (
                operation
                == "VISUAL_ATTRIBUTE_VERIFY"
            ):

                result = (
                    self._execute_visual_attribute_verify(
                        step=step,
                        step_outputs=step_outputs,
                        video_id=video_id,
                    )
                )

            # ------------------------------------------
            # Relationship filter
            # ------------------------------------------
            elif (
                operation
                == "RELATIONSHIP_FILTER"
            ):

                result = (
                    self._execute_relationship_filter(
                        step,
                        step_outputs,
                    )
                )
            #------------------------------------------
            # Action filter
            #------------------------------------------
            elif (
                operation
                == "ACTION_FILTER"
            ):

                result = (
                    self._execute_action_filter(
                        step=step,
                        step_outputs=step_outputs,
                    )
                )

            #-----------------------------------------
            #VLM filter
            #-----------------------------------------
            elif (
                operation
                == "VLM_VERIFY"
            ):

                result = (
                    self._execute_vlm_verify(
                        step=step,
                        step_outputs=step_outputs,
                        video_id=video_id,
                    )
                )


            elif (
                operation
                == "TEMPORAL_FILTER"
            ):

                result = (
                    self._execute_temporal_filter(
                        step=step,
                        step_outputs=step_outputs,
                    )
                )

            else:

                raise ValueError(
                    "Unsupported query plan "
                    f"operation: {operation}"
                )

            #
            # Store operation output.
            #

            if step.output:

                # Store by step ID for dependency lookup.
                step_outputs[
                    step.step_id
                ] = result

                # Also store by planner output name so the
                # final named output (usually "results")
                # can be retrieved directly.
                step_outputs[
                    step.output
                ] = result

        #
        # The planner's final output should
        # normally be called "results".
        #

        return step_outputs.get(
            "results",
            [],
        )

    # ==================================================
    # TRACK LOOKUP
    # ==================================================
    def _execute_track_lookup(
        self,
        step,
        video_id: str,
    ):

        entity_id = (
            step.params.get(
                "entity_id",
                step.step_id,
            )
        )

        label = (
            step.params["label"]
        )

        attributes = (
            step.params.get(
                "attributes",
                {},
            )
        )

        # ----------------------------------------------
        # Resolve query entity onto existing
        # SAM3 indexes.
        # ----------------------------------------------

        resolved = (
            self.entity_resolver.resolve(
                video_id=
                    video_id,

                entity_id=
                    entity_id,

                label=
                    label,

                attributes=
                    attributes,
            )
        )

        if not resolved.found:

            print(
                f'  "{label}" -> '
                f"no compatible index"
            )

            return []

        print(
            f'  Entity "{label}" resolved to:'
        )

        tracks = []

        seen_track_ids = set()

        for match in (
            resolved.matches
        ):

            index = (
                match.index
            )

            print(
                f"    {index.label}"
                f" | score={match.score:.3f}"
                f" | {match.reason}"
            )

            index_tracks = (
                self.track_store.load_tracks(
                    video_id=
                        video_id,

                    label=
                        index.label,
                )
            )

            #
            # Protect against duplicate tracks
            # if multiple indexes eventually
            # reference overlapping evidence.
            #

            for track in index_tracks:

                track_key = (
                    track.track_id
                )

                if (
                    track_key
                    in seen_track_ids
                ):
                    continue

                seen_track_ids.add(
                    track_key
                )

                tracks.append(
                    track
                )

        print(
            f"  -> {len(tracks)} track(s)"
        )

        return tracks

    # ==================================================
    # TEMPORAL OVERLAP
    # ==================================================

    def _execute_temporal_overlap(
        self,
        step,
        step_outputs,
    ):

        dependencies = [

            step_outputs.get(
                dependency,
                []
            )

            for dependency
            in step.depends_on
        ]

        if not dependencies:

            return []

        if any(
            not group
            for group
            in dependencies
        ):

            return []

        windows = []

        first_group = (
            dependencies[0]
        )

        for track in first_group:

            windows.append(
                CandidateWindow(
                    start_time=
                        track.start_time,

                    end_time=
                        track.end_time,

                    tracks=[
                        track
                    ],
                )
            )

        for group in dependencies[1:]:

            next_windows = []

            for window in windows:

                for track in group:

                    start = max(
                        window.start_time,
                        track.start_time,
                    )

                    end = min(
                        window.end_time,
                        track.end_time,
                    )

                    if end >= start:

                        next_windows.append(
                            CandidateWindow(
                                start_time=start,
                                end_time=end,

                                tracks=(
                                    window.tracks
                                    + [track]
                                ),
                            )
                        )

            windows = next_windows

            if not windows:
                break

        print(
            f"  overlap -> "
            f"{len(windows)} window(s)"
        )

        return windows

    # ==================================================
    # BUILD CANDIDATE WINDOWS
    # ==================================================

    def _execute_build_candidates(
        self,
        step,
        step_outputs,
    ):

        inputs = (
            self._get_dependency_values(
                step,
                step_outputs,
            )
        )

        if not inputs:
            return []

        source = inputs[0]

        padding_before = (
            step.params.get(
                "padding_before_seconds",
                0.0,
            )
        )

        padding_after = (
            step.params.get(
                "padding_after_seconds",
                0.0,
            )
        )

        candidates = []

        for item in source:

            #
            # Already a CandidateWindow
            #

            if isinstance(
                item,
                CandidateWindow,
            ):

                start = item.start_time
                end = item.end_time
                tracks = item.tracks

            #
            # Raw ObjectTrack
            #

            else:

                start = item.start_time
                end = item.end_time
                tracks = [item]

            start = max(
                0.0,
                start - padding_before,
            )

            end = (
                end
                + padding_after
            )

            candidates.append(
                CandidateWindow(
                    start_time=start,
                    end_time=end,
                    tracks=tracks,
                )
            )

        candidates = (
            self._merge_windows(
                candidates
            )
        )

        print(
            f"  built "
            f"{len(candidates)} candidate "
            f"window(s)"
        )

        return candidates

    # ==================================================
    # RANK RESULTS
    # ==================================================

    def _execute_rank_results(
        self,
        step,
        step_outputs,
    ):

        inputs = (
            self._get_dependency_values(
                step,
                step_outputs,
            )
        )

        if not inputs:
            return []

        windows = inputs[0]

        #
        # V0.4 ranking:
        #
        # More supporting tracks = stronger
        # candidate.
        #
        # We will replace this with a proper
        # evidence score later.
        #

        for window in windows:

            window.score = float(
                len(
                    window.tracks
                )
            )

        windows.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        print(
            f"  ranked "
            f"{len(windows)} result(s)"
        )

        return windows

    def _execute_relationship_filter(
            self,
            step,
            step_outputs,
        ):

            relationships = (
                step.params.get(
                    "relationships",
                    [],
                )
            )

            if not relationships:

                print(
                    "  no relationships"
                )

                return []

            #
            # First implementation:
            #
            # Execute one spatial relationship
            # correctly before introducing
            # multi-relationship conjunctions.
            #

            if len(relationships) > 1:

                raise NotImplementedError(
                    "Multiple simultaneous "
                    "relationships are not "
                    "implemented yet."
                )

            relationship = (
                relationships[0]
            )

            subject_id = (
                relationship[
                    "subject"
                ]
            )

            object_id = (
                relationship[
                    "object"
                ]
            )

            predicate = (
                relationship[
                    "predicate"
                ]
            )

            # ==============================================
            # GET ENTITY TRACKS
            # ==============================================

            subject_tracks = (
                self._get_entity_tracks(
                    entity_id=subject_id,
                    step_outputs=step_outputs,
                )
            )

            object_tracks = (
                self._get_entity_tracks(
                    entity_id=object_id,
                    step_outputs=step_outputs,
                )
            )

            print(
                f"  relationship: "
                f"{subject_id} "
                f"{predicate} "
                f"{object_id}"
            )

            print(
                f"  subject tracks: "
                f"{len(subject_tracks)}"
            )

            print(
                f"  object tracks: "
                f"{len(object_tracks)}"
            )

            if (
                not subject_tracks
                or
                not object_tracks
            ):

                return []

            windows = []

            # ==============================================
            # COMPARE TRACK PAIRS
            # ==============================================

            for subject_track in (
                subject_tracks
            ):

                for object_track in (
                    object_tracks
                ):

                    #
                    # Exact same track should never
                    # be compared with itself.
                    #

                    if (
                        subject_track.video_id
                        ==
                        object_track.video_id
                        and
                        subject_track.label
                        ==
                        object_track.label
                        and
                        subject_track.track_id
                        ==
                        object_track.track_id
                    ):

                        continue

                    result = (
                        self.spatial_filter.evaluate(
                            subject_track=
                                subject_track,

                            object_track=
                                object_track,

                            relationship=
                                predicate,
                        )
                    )

                    print(
                        f"    "
                        f"{subject_track.label}"
                        f" #{subject_track.track_id}"
                        f" {predicate} "
                        f"{object_track.label}"
                        f" #{object_track.track_id}"
                        f" -> {result.status}"
                        f" ({result.confidence:.3f})"
                    )

                    if (
                        result.status
                        != "MATCH"
                    ):

                        continue

                    if not (
                        result.matching_frames
                    ):

                        continue

                    #
                    # Convert matching spatial
                    # frames into CandidateWindows.
                    #

                    pair_windows = (
                        self._build_spatial_windows(
                            subject_track=
                                subject_track,

                            object_track=
                                object_track,

                            matching_frames=
                                result.matching_frames,
                        )
                    )

                    windows.extend(
                        pair_windows
                    )

            print(
                f"  relationship -> "
                f"{len(windows)} window(s)"
            )

            return windows

    # ==================================================
    # PASS THROUGH
    # ==================================================

    def _pass_through(
        self,
        step,
        step_outputs,
    ):

        inputs = (
            self._get_dependency_values(
                step,
                step_outputs,
            )
        )

        if not inputs:
            return []

        return inputs[0]

    # ==================================================
    # WINDOW MERGING
    # ==================================================

    def _merge_windows(
        self,
        windows: list[CandidateWindow],
    ):

        if not windows:
            return []

        windows = sorted(
            windows,
            key=lambda item:
                item.start_time,
        )

        merged = [
            windows[0]
        ]

        for current in windows[1:]:

            previous = merged[-1]

            #
            # Windows overlap.
            #

            if (
                current.start_time
                <= previous.end_time
            ):

                previous.end_time = max(
                    previous.end_time,
                    current.end_time,
                )

                existing_ids = {
                    track.track_id
                    for track
                    in previous.tracks
                }

                for track in (
                    current.tracks
                ):

                    if (
                        track.track_id
                        not in existing_ids
                    ):

                        previous.tracks.append(
                            track
                        )

                        existing_ids.add(
                            track.track_id
                        )

            else:

                merged.append(
                    current
                )

        return merged

    # ==================================================
    # DEPENDENCY HELPERS
    # ==================================================

    def _get_dependency_values(
        self,
        step,
        step_outputs,
    ):

        return [

            step_outputs.get(
                dependency,
                []
            )

            for dependency
            in step.depends_on
        ]

    def _dependency_outputs(
        self,
        step,
        step_outputs,
    ):

        outputs = []

        for dependency in (
            step.depends_on
        ):

            for key in step_outputs:

                if (
                    dependency in key
                    or key in dependency
                ):
                    outputs.append(
                        key
                    )

        return outputs

    def _find_dependency_value(
        self,
        dependency,
        step_outputs,
    ):

        #
        # Direct lookup first.
        #

        if dependency in step_outputs:

            return step_outputs[
                dependency
            ]

        #
        # Planner output conventions.
        #

        mappings = {

            "entity_overlap":
                "overlap_windows",

            "relationship_filter":
                "relationship_windows",

            "action_filter":
                "action_windows",

            "temporal_filter":
                "temporal_windows",

            "build_candidates":
                "candidate_windows",

            "vlm_verify":
                "verified_windows",
        }

        if dependency in mappings:

            return step_outputs.get(
                mappings[dependency]
            )

        #
        # Entity lookup convention:
        #
        # find_entity_0
        #       ↓
        # entity_0_tracks
        #

        if dependency.startswith(
            "find_"
        ):

            entity_id = (
                dependency[
                    len("find_"):
                ]
            )

            return step_outputs.get(
                f"{entity_id}_tracks"
            )

        return None

    def _execute_attribute_filter(
        self,
        step,
        step_outputs,
    ):

        inputs = (
            self._get_dependency_values(
                step,
                step_outputs,
            )
        )

        if not inputs:
            return AttributeFilterResult()

        tracks = inputs[0]

        attributes = (
            step.params.get(
                "attributes",
                {},
            )
        )

        result = (
            self.attribute_filter.filter_tracks(
                tracks=tracks,
                attributes=attributes,
            )
        )

        print(
            f"  attributes -> "
            f"{len(result.verified)} verified, "
            f"{len(result.rejected)} rejected, "
            f"{len(result.unverified)} unverified"
        )

        for track in result.rejected:

            print(
                f'    REJECT "{track.label}"'
            )

        for track in result.unverified:

            print(
                f'    UNVERIFIED "{track.label}"'
            )

        return result



    def _execute_visual_attribute_verify(
    self,
    step,
    step_outputs,
    video_id: str,
    ):

        inputs = (
            self._get_dependency_values(
                step,
                step_outputs,
            )
        )

        if not inputs:
            return []

        state = inputs[0]

        if not isinstance(
            state,
            AttributeFilterResult,
        ):

            raise TypeError(
                "VISUAL_ATTRIBUTE_VERIFY "
                "expected AttributeFilterResult"
            )

        #
        # Tracks already proven by metadata /
        # index labels do not need Qwen.
        #

        verified_tracks = list(
            state.verified
        )

        #
        # Nothing uncertain.
        #

        if not state.unverified:

            print(
                "  no visual verification "
                "required"
            )

            return verified_tracks

        #
        # If no verifier is configured,
        # do NOT treat uncertain tracks as
        # matches.
        #

        if self.attribute_verifier is None:

            print(
                "  visual verifier unavailable"
            )

            return verified_tracks

        video_path = (
            self.track_store.get_video_path(
                video_id
            )
        )

        if not video_path:

            print(
                "  video path not found "
                "in track index"
            )

            return verified_tracks

        label = (
            step.params["label"]
        )

        attributes = (
            step.params.get(
                "attributes",
                {},
            )
        )

        for track in state.unverified:

            result = (
                self.attribute_verifier.verify_track(
                    video_path=
                        video_path,

                    video_id=
                        video_id,

                    track=
                        track,

                    entity_label=
                        label,

                    attributes=
                        attributes,
                )
            )

            print(
                f'    VLM "{track.label}"'
                f" -> {result.status}"
                f" ({result.confidence:.3f})"
                f" | {result.reason}"
            )

            if (
                result.status == "MATCH"
            ):

                verified_tracks.append(
                    track
                )

        print(
            f"  -> {len(verified_tracks)} "
            f"verified track(s)"
        )

        return verified_tracks


    def _execute_relationship_filter(
            self,
            step,
            step_outputs,
        ):

            relationships = (
                step.params.get(
                    "relationships",
                    [],
                )
            )

            if not relationships:

                print(
                    "  no relationships"
                )

                return []

            #
            # First implementation:
            #
            # Execute one spatial relationship
            # correctly before introducing
            # multi-relationship conjunctions.
            #

            if len(relationships) > 1:

                raise NotImplementedError(
                    "Multiple simultaneous "
                    "relationships are not "
                    "implemented yet."
                )

            relationship = (
                relationships[0]
            )

            subject_id = (
                relationship[
                    "subject"
                ]
            )

            object_id = (
                relationship[
                    "object"
                ]
            )

            predicate = (
                relationship[
                    "predicate"
                ]
            )

            # ==============================================
            # GET ENTITY TRACKS
            # ==============================================

            subject_tracks = (
                self._get_entity_tracks(
                    entity_id=subject_id,
                    step_outputs=step_outputs,
                )
            )

            object_tracks = (
                self._get_entity_tracks(
                    entity_id=object_id,
                    step_outputs=step_outputs,
                )
            )

            print(
                f"  relationship: "
                f"{subject_id} "
                f"{predicate} "
                f"{object_id}"
            )

            print(
                f"  subject tracks: "
                f"{len(subject_tracks)}"
            )

            print(
                f"  object tracks: "
                f"{len(object_tracks)}"
            )

            if (
                not subject_tracks
                or
                not object_tracks
            ):

                return []

            windows = []

            # ==============================================
            # COMPARE TRACK PAIRS
            # ==============================================

            for subject_track in (
                subject_tracks
            ):

                for object_track in (
                    object_tracks
                ):

                    #
                    # Exact same track should never
                    # be compared with itself.
                    #

                    if (
                        subject_track.video_id
                        ==
                        object_track.video_id
                        and
                        subject_track.label
                        ==
                        object_track.label
                        and
                        subject_track.track_id
                        ==
                        object_track.track_id
                    ):

                        continue

                    result = (
                        self.spatial_filter.evaluate(
                            subject_track=
                                subject_track,

                            object_track=
                                object_track,

                            relationship=
                                predicate,
                        )
                    )

                    print(
                        f"    "
                        f"{subject_track.label}"
                        f" #{subject_track.track_id}"
                        f" {predicate} "
                        f"{object_track.label}"
                        f" #{object_track.track_id}"
                        f" -> {result.status}"
                        f" ({result.confidence:.3f})"
                    )

                    if (
                        result.status
                        != "MATCH"
                    ):

                        continue

                    if not (
                        result.matching_frames
                    ):

                        continue

                    #
                    # Convert matching spatial
                    # frames into CandidateWindows.
                    #

                    pair_windows = (
                        self._build_spatial_windows(
                            subject_track=
                                subject_track,

                            object_track=
                                object_track,

                            matching_frames=
                                result.matching_frames,
                        )
                    )

                    windows.extend(
                        pair_windows
                    )

            print(
                f"  relationship -> "
                f"{len(windows)} window(s)"
            )

            return windows


    def _build_spatial_windows(
            self,
            subject_track,
            object_track,
            matching_frames,
        ):

            if not matching_frames:

                return []

            matching_frames = sorted(
                matching_frames
            )

            #
            # Split:
            #
            # [1,2,3,8,9]
            #
            # into:
            #
            # [1,2,3]
            # [8,9]
            #

            runs = []

            current_run = [
                matching_frames[0]
            ]

            for frame_index in (
                matching_frames[1:]
            ):

                previous_frame = (
                    current_run[-1]
                )

                if (
                    frame_index
                    ==
                    previous_frame + 1
                ):

                    current_run.append(
                        frame_index
                    )

                else:

                    runs.append(
                        current_run
                    )

                    current_run = [
                        frame_index
                    ]

            runs.append(
                current_run
            )

            # ==============================================
            # FRAME → TIMESTAMP LOOKUP
            # ==============================================

            subject_points = {

                point.frame_index:
                    point

                for point
                in subject_track.points
            }

            object_points = {

                point.frame_index:
                    point

                for point
                in object_track.points
            }

            windows = []

            for run in runs:

                timestamps = []

                for frame_index in run:

                    subject_point = (
                        subject_points.get(
                            frame_index
                        )
                    )

                    object_point = (
                        object_points.get(
                            frame_index
                        )
                    )

                    if (
                        subject_point
                        is not None
                    ):

                        timestamps.append(
                            subject_point.timestamp
                        )

                    elif (
                        object_point
                        is not None
                    ):

                        timestamps.append(
                            object_point.timestamp
                        )

                if not timestamps:

                    continue

                windows.append(
                    CandidateWindow(
                        start_time=min(
                            timestamps
                        ),

                        end_time=max(
                            timestamps
                        ),

                        tracks=[
                            subject_track,
                            object_track,
                        ],
                    )
                )

            return windows



    def _get_entity_tracks(
            self,
            entity_id: str,
            step_outputs,
        ):

            #
            # Prefer visually verified tracks
            # when attributes were present.
            #

            verified_key = (
                f"{entity_id}"
                "_verified_tracks"
            )

            if verified_key in step_outputs:

                return step_outputs[
                    verified_key
                ]

            #
            # Otherwise use normal track
            # lookup output.
            #

            track_key = (
                f"{entity_id}_tracks"
            )

            if track_key in step_outputs:

                return step_outputs[
                    track_key
                ]

            return []



    def _track_key(
        self,
        track,
    ):

        return (
            track.video_id,
            track.label,
            track.track_id,
        )

    def _execute_action_filter(
        self,
        step,
        step_outputs,
    ):

        inputs = (
            self._get_dependency_values(
                step,
                step_outputs,
            )
        )

        if not inputs:

            return []

        source = inputs[0]

        actions = (
            step.params.get(
                "actions",
                [],
            )
        )

        if not actions:

            return source

        # ==============================================
        # NORMALIZE EVERYTHING TO WINDOWS
        # ==============================================

        windows = []

        for item in source:

            if isinstance(
                item,
                CandidateWindow,
            ):

                windows.append(
                    item
                )

            else:

                #
                # Raw ObjectTrack.
                #
                # This happens for queries such as:
                #
                # "person walking"
                #
                # where there may only be one entity.
                #

                windows.append(
                    CandidateWindow(
                        start_time=
                            item.start_time,

                        end_time=
                            item.end_time,

                        tracks=[
                            item
                        ],
                    )
                )

        filtered = []

        # ==============================================
        # PARTICIPANT PREFILTER
        # ==============================================

        for window in windows:

            if self._window_supports_actions(
                window=window,
                actions=actions,
                step_outputs=step_outputs,
            ):

                filtered.append(
                    window
                )

        print(
            f"  action candidates -> "
            f"{len(filtered)} / "
            f"{len(windows)} window(s)"
        )

        return filtered







    def _window_supports_actions(
        self,
        window,
        actions,
        step_outputs,
    ):

        window_track_keys = {

            self._track_key(
                track
            )

            for track
            in window.tracks
        }

        for action in actions:

            #
            # ActionSpec:
            #
            # actor
            # object
            # target
            #

            participant_ids = [

                action.get(
                    "actor"
                ),

                action.get(
                    "object"
                ),

                action.get(
                    "target"
                ),
            ]

            participant_ids = [

                participant_id

                for participant_id
                in participant_ids

                if participant_id
            ]

            for entity_id in (
                participant_ids
            ):

                entity_tracks = (
                    self._get_entity_tracks(
                        entity_id=
                            entity_id,

                        step_outputs=
                            step_outputs,
                    )
                )

                if not entity_tracks:

                    print(
                        f"    action participant "
                        f"'{entity_id}' "
                        f"has no tracks"
                    )

                    return False

                entity_track_keys = {

                    self._track_key(
                        track
                    )

                    for track
                    in entity_tracks
                }

                #
                # Window must contain at least
                # one track belonging to this
                # action participant.
                #

                if (
                    window_track_keys
                    .isdisjoint(
                        entity_track_keys
                    )
                ):

                    return False

        return True

    def _execute_vlm_verify(
        self,
        step,
        step_outputs,
        video_id: str,
    ):

        inputs = (
            self._get_dependency_values(
                step,
                step_outputs,
            )
        )

        if not inputs:

            return []

        windows = inputs[0]

        if not windows:

            return []

        query = (
            step.params[
                "query"
            ]
        )

        required = (
            step.params.get(
                "required",
                False,
            )
        )

        # ==============================================
        # VERIFIER AVAILABILITY
        # ==============================================

        if (
            self.video_verifier
            is None
        ):

            print(
                "  video verifier unavailable"
            )

            #
            # Actions MUST NOT silently pass.
            #

            if required:

                return []

            #
            # Relationship-only queries have
            # already been geometrically verified.
            #

            return windows

        video_path = (
            self.track_store
            .get_video_path(
                video_id
            )
        )

        if not video_path:

            print(
                "  video path not found"
            )

            return []

        verified_windows = []

        # ==============================================
        # VERIFY EACH WINDOW
        # ==============================================

        for index, window in enumerate(
            windows
        ):

            with tempfile.TemporaryDirectory(
                prefix=(
                    "video_query_"
                )
            ) as temp_dir:

                frame_paths = (
                    self.window_frame_sampler
                    .sample(
                        video_path=
                            video_path,

                        start_time=
                            window.start_time,

                        end_time=
                            window.end_time,

                        output_dir=
                            temp_dir,
                    )
                )

                if not frame_paths:

                    print(
                        f"    window #{index}: "
                        f"no frames"
                    )

                    continue

                segment = {

                    "start_time":
                        window.start_time,

                    "end_time":
                        window.end_time,

                    "frame_paths":
                        frame_paths,
                }

                result = (
                    self.video_verifier
                    .verify(
                        query=query,
                        segment=segment,
                    )
                )

                print(
                    f"    window "
                    f"{window.start_time:.2f}s"
                    f" -> "
                    f"{window.end_time:.2f}s"
                    f" | "
                    f"{'MATCH' if result.match else 'REJECT'}"
                    f" ({result.confidence:.3f})"
                    f" | {result.reason}"
                )

                if not result.match:

                    continue

                #
                # Keep useful evidence on
                # the CandidateWindow.
                #

                window.vlm_confidence = (
                    result.confidence
                )

                window.vlm_reason = (
                    result.reason
                )

                window.vlm_evidence_frames = (
                    result.evidence_frames
                )

                verified_windows.append(
                    window
                )

        print(
            f"  VLM -> "
            f"{len(verified_windows)} / "
            f"{len(windows)} "
            f"verified window(s)"
        )

        return verified_windows




    def _get_temporal_entity_windows(
        self,
        entity_id,
        step_outputs,
    ):

        tracks = (
            self._get_entity_tracks(
                entity_id=entity_id,
                step_outputs=step_outputs,
            )
        )

        return [
            CandidateWindow(
                start_time=
                    track.start_time,

                end_time=
                    track.end_time,

                tracks=[
                    track
                ],
            )

            for track
            in tracks
        ]



    def _execute_temporal_filter(
        self,
        step,
        step_outputs,
    ):

        constraints = (
            step.params.get(
                "constraints",
                [],
            )
        )

        inputs = (
            self._get_dependency_values(
                step,
                step_outputs,
            )
        )

        if not inputs:

            return []

        source_windows = (
            inputs[0]
        )

        if not constraints:

            return source_windows

        current_windows = list(
            source_windows
        )

        for constraint in constraints:

            relation = (
                constraint[
                    "relation"
                ]
            )

            first_id = (
                constraint[
                    "first"
                ]
            )

            second_id = (
                constraint[
                    "second"
                ]
            )

            value = (
                constraint.get(
                    "value"
                )
            )

            unit = (
                constraint.get(
                    "unit"
                )
            )

            first_windows = (
                self._get_temporal_entity_windows(
                    entity_id=first_id,
                    step_outputs=step_outputs,
                )
            )

            second_windows = (
                self._get_temporal_entity_windows(
                    entity_id=second_id,
                    step_outputs=step_outputs,
                )
            )

            #
            # If either reference is not an
            # entity, it may refer to an action.
            #
            # We defer those cases to Qwen for
            # now rather than pretending that
            # an action has the same duration
            # as its actor's entire track.
            #

            if (
                not first_windows
                or
                not second_windows
            ):

                print(
                    f"  temporal '{relation}' "
                    f"requires event/action "
                    f"reasoning -> deferred to VLM"
                )

                continue

            matching_keys = set()

            print(
                f"  temporal: "
                f"{first_id} "
                f"{relation} "
                f"{second_id}"
            )

            for first in first_windows:

                for second in second_windows:

                    result = (
                        self.temporal_filter
                        .evaluate(
                            first=first,
                            second=second,
                            relation=relation,
                            value=value,
                            unit=unit,
                        )
                    )

                    first_track = (
                        first.tracks[0]
                    )

                    second_track = (
                        second.tracks[0]
                    )

                    print(
                        f"    "
                        f"{first_track.label}"
                        f" #{first_track.track_id}"
                        f" {relation} "
                        f"{second_track.label}"
                        f" #{second_track.track_id}"
                        f" -> {result.status}"
                    )

                    if (
                        result.status
                        !=
                        "MATCH"
                    ):
                        continue

                    matching_keys.add(
                        self._track_key(
                            first_track
                        )
                    )

                    matching_keys.add(
                        self._track_key(
                            second_track
                        )
                    )

            if not matching_keys:

                current_windows = []
                break

            #
            # Keep candidate windows containing
            # evidence satisfying the constraint.
            #

            filtered = []

            for window in (
                current_windows
            ):

                window_keys = {

                    self._track_key(
                        track
                    )

                    for track
                    in window.tracks
                }

                if (
                    window_keys
                    &
                    matching_keys
                ):

                    filtered.append(
                        window
                    )

            current_windows = (
                filtered
            )

        print(
            f"  temporal -> "
            f"{len(current_windows)} "
            f"window(s)"
        )

        return current_windows