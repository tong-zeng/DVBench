from PIL import Image
from io import BytesIO
import base64


def process_image(image, image_process_mode="Default", return_pil=False, image_format='PNG', max_len=1344, min_len=672):
    if image_process_mode == "Pad":
        def expand2square(pil_img, background_color=(122, 116, 104)):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image)
    elif image_process_mode in ["Default", "Crop"]:
        pass
    elif image_process_mode == "Resize":
        image = image.resize((336, 336))
    else:
        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
    if max(image.size) > max_len:
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
    if return_pil:
        return image
    else:
        buffered = BytesIO()
        image.save(buffered, format=image_format)
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        return img_b64_str
