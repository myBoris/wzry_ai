import cv2
import os
import numpy as np


class TemplateMatcher:
    def __init__(self, template_dir, threshold=0.8):
        self.templates = []
        self.template_names = []
        self.threshold = threshold
        self.load_templates(template_dir)

    def load_templates(self, template_dir):
        for template_file in os.listdir(template_dir):
            template_path = os.path.join(template_dir, template_file)
            template = cv2.imread(template_path, 0)  # Ensure template is read as grayscale
            self.templates.append(template)
            self.template_names.append(os.path.splitext(template_file)[0])

    def crop_to_circle(self, image):
        x = 745
        y = 20
        r = 15

        height, width = image.shape[:2]

        # Create a mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, (255), -1)

        # Apply the mask to the image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Create an alpha channel based on the mask
        if image.shape[2] == 3:
            # If image has no alpha channel, add one
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask

        # Crop the image to the bounding box of the circle
        result = result[y - r:y + r, x - r:x + r]

        return result

    def multi_scale_template_matching(self, image, template, scales):
        best_match_score = -1
        best_scale = None
        best_location = None

        for scale in scales:
            resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if image.shape[0] < resized_template.shape[0] or image.shape[1] < resized_template.shape[1]:
                continue
            res = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_match_score:
                best_match_score = max_val
                best_scale = scale
                best_location = max_loc

        return best_match_score, best_scale, best_location

    def match_template(self, image):
        # Crop the image
        cropped_image = self.crop_to_circle(image)

        # Ensure the cropped image is grayscale
        if cropped_image.shape[2] == 4:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2GRAY)
        elif cropped_image.shape[2] == 3:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Initialize the best match result
        best_match_score = -1
        best_template_name = None

        # Perform template matching for each template
        for template, template_name in zip(self.templates, self.template_names):
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # Adjustable scale range
            match_score, _, _ = self.multi_scale_template_matching(cropped_image, template, scales)

            if match_score > best_match_score:
                best_match_score = match_score
                best_template_name = template_name

        # Return the best matching template name if the score is above the threshold
        if best_match_score >= self.threshold:
            return best_template_name
        else:
            return None


# Usage example
template_dir = "template"
image_path = "H:\\video\\monster\\src\\some_image.png"
matcher = TemplateMatcher(template_dir)

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
matched_template = matcher.match_template(image)

print(f"Matched Template: {matched_template}")
