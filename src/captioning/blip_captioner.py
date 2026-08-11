import torch

from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
)


class BLIPCaptioner:

    def __init__(
        self,
        model_name: str = (
            "Salesforce/"
            "blip-image-captioning-base"
        ),
        device: str = "cpu",
    ):

        self.device = torch.device(
            device
        )

        print(
            f"Loading caption model: "
            f"{model_name}"
        )

        self.processor = (
            BlipProcessor
            .from_pretrained(
                model_name
            )
        )

        self.model = (
            BlipForConditionalGeneration
            .from_pretrained(
                model_name
            )
            .to(self.device)
        )

        self.model.eval()

    def caption(
        self,
        image_path: str,
        max_new_tokens: int = 40,
    ) -> str:

        image = (
            Image
            .open(image_path)
            .convert("RGB")
        )

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            output_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=
                        max_new_tokens,
                )
            )

        caption = (
            self.processor.decode(
                output_ids[0],
                skip_special_tokens=True,
            )
            .strip()
        )

        image.close()

        return caption