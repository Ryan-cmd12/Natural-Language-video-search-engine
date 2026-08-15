from src.query.plan_models import (
    PlanStep,
    QueryPlan,
)


class QueryPlanner:
    

    def plan(
        self,
        compiled_query,
    ) -> QueryPlan:

        plan = QueryPlan(
            query=self._get_query_text(
                compiled_query
            )
        )

        entities = (
            getattr(
                compiled_query,
                "entities",
                None,
            )
            or []
        )

        actions = (
            getattr(
                compiled_query,
                "actions",
                None,
            )
            or []
        )

        relationships = (
            getattr(
                compiled_query,
                "relationships",
                None,
            )
            or []
        )

        temporal_constraints = (
            getattr(
                compiled_query,
                "temporal_constraints",
                None,
            )
            or []
        )
        if temporal_constraints:

            raise NotImplementedError(
                "Temporal constraints are "
                "not supported in V0.4."
            )

        entity_steps = []

        for index, entity in enumerate(
            entities
        ):

            entity_id = (
                self._get_entity_id(
                    entity,
                    index,
                )
            )

            label = (
                self._get_entity_label(
                    entity
                )
            )

            attributes = (
                self._get_entity_attributes(
                    entity
                )
            )

            # ==============================================
            # ENTITY LOOKUP
            # ==============================================

            lookup_step_id = (
                f"find_{entity_id}"
            )

            plan.steps.append(
                PlanStep(
                    step_id=
                        lookup_step_id,

                    operation=
                        "TRACK_LOOKUP",

                    description=(
                        f'Find tracks matching '
                        f'"{label}"'
                    ),

                    params={
                        "entity_id":
                            entity_id,

                        "label":
                            label,

                        "attributes":
                            attributes,
                    },

                    output=(
                        f"{entity_id}_tracks"
                    ),
                )
            )

            #
            # By default the entity's final
            # evidence comes directly from lookup.
            #

            entity_step_id = (
                lookup_step_id
            )

            # ==============================================
            # ATTRIBUTE FILTER
            # ==============================================

            if attributes:
                # ==============================================
                # CHEAP ATTRIBUTE FILTER
                # ==============================================

                attribute_step_id = (
                    f"attributes_{entity_id}"
                )

                plan.steps.append(
                    PlanStep(
                        step_id=
                            attribute_step_id,

                        operation=
                            "ATTRIBUTE_FILTER",

                        description=(
                            f"Filter {label} tracks "
                            f"using indexed attributes"
                        ),

                        depends_on=[
                            lookup_step_id
                        ],

                        params={
                            "entity_id":
                                entity_id,

                            "label":
                                label,

                            "attributes":
                                attributes,
                        },

                        output=(
                            f"{entity_id}"
                            "_attribute_state"
                        ),
                    )
                )

                # ==============================================
                # VLM ATTRIBUTE VERIFICATION
                # ==============================================

                verify_step_id = (
                    f"verify_attributes_"
                    f"{entity_id}"
                )

                plan.steps.append(
                    PlanStep(
                        step_id=
                            verify_step_id,

                        operation=
                            "VISUAL_ATTRIBUTE_VERIFY",

                        description=(
                            f"Visually verify unresolved "
                            f"attributes for {label}"
                        ),

                        depends_on=[
                            attribute_step_id
                        ],

                        params={
                            "entity_id":
                                entity_id,

                            "label":
                                label,

                            "attributes":
                                attributes,
                        },

                        output=(
                            f"{entity_id}"
                            "_verified_tracks"
                        ),
                    )
                )

                entity_step_id = (
                    verify_step_id
                )

            #
            # IMPORTANT:
            #
            # Temporal overlap now depends on the
            # FILTERED entity result.
            #

            entity_steps.append(
                entity_step_id
            )

        #
        # STEP 2:
        # If multiple entities exist,
        # find temporal overlap.
        #

        candidate_dependency = (
            entity_steps.copy()
        )

        if len(entity_steps) > 1:

            plan.steps.append(
                PlanStep(
                    step_id=
                    "entity_overlap",

                    operation=
                    "TEMPORAL_OVERLAP",

                    description=(
                        "Find time windows where "
                        "the required entity tracks "
                        "co-exist"
                    ),

                    depends_on=
                    entity_steps,

                    params={
                        "minimum_overlap_seconds":
                            0.0,
                    },

                    output=
                    "overlap_windows",
                )
            )

            candidate_dependency = [
                "entity_overlap"
            ]

        #
        # STEP 3:
        # Relationship reasoning
        #

        if relationships:

            plan.steps.append(
                PlanStep(
                    step_id=
                    "relationship_filter",

                    operation=
                    "RELATIONSHIP_FILTER",

                    description=(
                        "Filter candidate windows "
                        "using spatial/object "
                        "relationships"
                    ),

                    depends_on=
                    candidate_dependency,

                    params={
                        "relationships": [
                            self._object_to_dict(
                                relationship
                            )
                            for relationship
                            in relationships
                        ]
                    },

                    output=
                    "relationship_windows",
                )
            )

            candidate_dependency = [
                "relationship_filter"
            ]

        #
        # STEP 4:
        # Action/event reasoning
        #

        if actions:

            plan.steps.append(
                PlanStep(
                    step_id=
                    "action_filter",

                    operation=
                    "ACTION_FILTER",

                    description=(
                        "Check candidate windows "
                        "for the requested actions"
                    ),

                    depends_on=
                    candidate_dependency,

                    params={
                        "actions": [
                            self._object_to_dict(
                                action
                            )
                            for action
                            in actions
                        ]
                    },

                    output=
                    "action_windows",
                )
            )

            candidate_dependency = [
                "action_filter"
            ]

            #
            # Actions usually require
            # visual temporal reasoning.
            #

            plan.requires_vlm_verification = (
                True
            )

        #
        # STEP 5:
        # Temporal constraints
        #

        if temporal_constraints:

            plan.steps.append(
                PlanStep(
                    step_id=
                    "temporal_filter",

                    operation=
                    "TEMPORAL_FILTER",

                    description=(
                        "Apply temporal constraints "
                        "to candidate windows"
                    ),

                    depends_on=
                    candidate_dependency,

                    params={
                        "constraints": [
                            self._object_to_dict(
                                constraint
                            )
                            for constraint
                            in temporal_constraints
                        ]
                    },

                    output=
                    "temporal_windows",
                )
            )

            candidate_dependency = [
                "temporal_filter"
            ]

        #
        # STEP 6:
        # Convert results into candidate
        # video windows.
        #

        plan.steps.append(
            PlanStep(
                step_id=
                "build_candidates",

                operation=
                "BUILD_CANDIDATE_WINDOWS",

                description=(
                    "Convert matching evidence "
                    "into video timestamp windows"
                ),

                depends_on=
                candidate_dependency,

                params={
                    "padding_before_seconds":
                        1.0,

                    "padding_after_seconds":
                        1.0,
                },

                output=
                "candidate_windows",
            )
        )

        #
        # STEP 7:
        # VLM verification when needed.
        #

        if (
            plan.requires_vlm_verification
            or relationships
        ):

            plan.requires_vlm_verification = (
                True
            )

            plan.steps.append(
                PlanStep(
                    step_id=
                    "vlm_verify",

                    operation=
                    "VLM_VERIFY",

                    description=(
                        "Use Qwen VLM to verify "
                        "candidate windows"
                    ),

                    depends_on=[
                        "build_candidates"
                    ],

                    params={
                        "query":
                            plan.query,

                        #
                        # Action semantics cannot be
                        # safely accepted without VLM.
                        #

                        "required":
                            bool(actions),
                    },

                    output=
                    "verified_windows",
                )
            )

            ranking_dependency = [
                "vlm_verify"
            ]

        else:

            ranking_dependency = [
                "build_candidates"
            ]

        #
        # FINAL STEP
        #

        plan.steps.append(
            PlanStep(
                step_id=
                "rank_results",

                operation=
                "RANK_RESULTS",

                description=(
                    "Rank matching timestamp "
                    "windows"
                ),

                depends_on=
                ranking_dependency,

                output=
                "results",
            )
        )

        return plan

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _get_query_text(
        self,
        compiled_query,
    ):

        for field in [
            "query",
            "original_query",
            "text",
        ]:

            value = getattr(
                compiled_query,
                field,
                None,
            )

            if value:
                return value

        return ""

    def _get_entity_id(
        self,
        entity,
        index,
    ):

        for field in [
            "entity_id",
            "id",
            "ref",
        ]:

            value = getattr(
                entity,
                field,
                None,
            )

            if value:
                return str(value)

        return f"entity_{index}"

    def _get_entity_label(
        self,
        entity,
    ):

        for field in [
            "concept",
            "type",    
            "label",
            "name",
            "description",
            "text",
        ]:

            value = getattr(
                entity,
                field,
                None,
            )

            if value:
                return str(value)

        return str(entity)
    def _get_entity_attributes(
        self,
        entity,
    ):

        attributes = getattr(
            entity,
            "attributes",
            None,
        )

        if attributes is None:
            return []

        return attributes

    def _object_to_dict(
        self,
        obj,
    ):

        if hasattr(
            obj,
            "model_dump",
        ):
            return obj.model_dump()

        if hasattr(
            obj,
            "to_dict",
        ):
            return obj.to_dict()

        if hasattr(
            obj,
            "__dict__",
        ):
            return dict(
                obj.__dict__
            )

        return {
            "value": str(obj)
        }


    