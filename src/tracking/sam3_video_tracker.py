from sam3.model_builder import (
    build_sam3_video_predictor,
)


class Sam3VideoTracker:

    def __init__(self):

        print(
            "Loading SAM 3 video predictor..."
        )

        self.predictor = (
            build_sam3_video_predictor()
        )

    def start_session(
        self,
        resource_path: str,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> str:

        response = (
            self.predictor.handle_request(
                request={
                    "type": "start_session",

                    "resource_path":
                        resource_path,

                    "offload_video_to_cpu":
                        offload_video_to_cpu,

                    "offload_state_to_cpu":
                        offload_state_to_cpu,
                }
            )
        )

        return response[
            "session_id"
        ]

    def add_text_prompt(
        self,
        session_id: str,
        frame_index: int,
        text: str,
        output_prob_thresh: float = 0.5,
    ) -> dict:

        response = (
            self.predictor.handle_request(
                request={
                    "type": "add_prompt",

                    "session_id":
                        session_id,

                    "frame_index":
                        frame_index,

                    "text":
                        text,

                    "output_prob_thresh":
                        output_prob_thresh,
                }
            )
        )

        return response

    def propagate(
        self,
        session_id: str,
        direction: str = "forward",
        start_frame_index: int | None = None,
        max_frames: int | None = None,
        output_prob_thresh: float = 0.5,
    ):

        request = {
            "type":
                "propagate_in_video",

            "session_id":
                session_id,

            "propagation_direction":
                direction,

            "output_prob_thresh":
                output_prob_thresh,
        }

        if start_frame_index is not None:
            request[
                "start_frame_index"
            ] = start_frame_index

        if max_frames is not None:
            request[
                "max_frame_num_to_track"
            ] = max_frames

        yield from (
            self.predictor
            .handle_stream_request(
                request=request
            )
        )

    def close_session(
        self,
        session_id: str,
    ):

        return (
            self.predictor.handle_request(
                request={
                    "type":
                        "close_session",

                    "session_id":
                        session_id,
                }
            )
        )