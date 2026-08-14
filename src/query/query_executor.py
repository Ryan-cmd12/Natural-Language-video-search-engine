from src.query.result_models import (
    CandidateWindow,
)

from src.storage.track_store import (
    TrackStore,
)

from src.query.entity_resolver import (
    EntityResolver,
)

class QueryExecutor:

    def __init__(
        self,
        track_store: TrackStore,
    ):

        self.track_store = (
            track_store
        )

        self.entity_resolver = (
            EntityResolver(
                track_store=
                    track_store
            )
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

            # ------------------------------------------
            # NOT IMPLEMENTED YET
            # ------------------------------------------

            elif operation in {
                "RELATIONSHIP_FILTER",
                "ACTION_FILTER",
                "TEMPORAL_FILTER",
                "VLM_VERIFY",
            }:

                result = (
                    self._pass_through(
                        step,
                        step_outputs,
                    )
                )

                print(
                    "  currently pass-through"
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