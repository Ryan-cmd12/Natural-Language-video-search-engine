from sam3.model_builder import build_sam3_video_predictor


class Sam3VideoTracker:
    def __init__(self):
        print("Loading SAM 3 video predictor...")
        self.predictor = build_sam3_video_predictor()

    def start_session(
        self,
        resource_path: str,
    ) -> str:
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=resource_path,
            )
        )

        return response["session_id"]

    def add_text_prompt(
        self,
        session_id: str,
        frame_index: int,
        text: str,
    ) -> dict:
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=text,
            )
        )

        return response["outputs"]