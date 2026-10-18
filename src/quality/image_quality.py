import os
import sys
import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

TARGET_SIZE = (224, 224)


# ============================================================
# MRI QUALITY ASSESSMENT
# ============================================================

def assess_mri_quality(image_input):

    """
    Heuristic MRI image quality assessment.

    Checks:
    1. Blur / sharpness
    2. Brightness
    3. Contrast
    4. Resolution

    Returns a dictionary containing scores and overall status.
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    if isinstance(image_input, str):

        image = cv2.imread(image_input)

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_input}"
            )

    else:

        image = np.array(image_input)

        # PIL / RGB -> OpenCV BGR
        if len(image.shape) == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )


    original_height, original_width = image.shape[:2]


    # --------------------------------------------------------
    # Resize to standard size for consistent measurement
    # --------------------------------------------------------

    resized = cv2.resize(
        image,
        TARGET_SIZE,
        interpolation=cv2.INTER_LINEAR
    )


    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # 1. SHARPNESS / BLUR
    # ========================================================

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


    if blur_score < 40:

        blur_status = "Poor"

    elif blur_score < 80:

        blur_status = "Acceptable"

    else:

        blur_status = "Good"


    # ========================================================
    # 2. FOREGROUND MASK
    # ========================================================

    # Ignore almost-black MRI background.

    foreground_pixels = gray[
        gray > 10
    ]


    if foreground_pixels.size < 100:

        foreground_pixels = gray.flatten()


    # ========================================================
    # 3. BRIGHTNESS
    # ========================================================

    brightness_score = float(
        np.mean(
            foreground_pixels
        )
    )


    if (
        brightness_score < 25
        or
        brightness_score > 235
    ):

        brightness_status = "Poor"

    elif (
        brightness_score < 40
        or
        brightness_score > 220
    ):

        brightness_status = "Acceptable"

    else:

        brightness_status = "Good"


    # ========================================================
    # 4. CONTRAST
    # ========================================================

    contrast_score = float(
        np.std(
            foreground_pixels
        )
    )


    if contrast_score < 18:

        contrast_status = "Poor"

    elif contrast_score < 28:

        contrast_status = "Acceptable"

    else:

        contrast_status = "Good"


    # ========================================================
    # 5. RESOLUTION
    # ========================================================

    minimum_dimension = min(
        original_width,
        original_height
    )


    if minimum_dimension < 128:

        resolution_status = "Poor"

    elif minimum_dimension < 224:

        resolution_status = "Acceptable"

    else:

        resolution_status = "Good"


    # ========================================================
    # OVERALL QUALITY
    # ========================================================

    statuses = [
        blur_status,
        brightness_status,
        contrast_status,
        resolution_status
    ]


    if "Poor" in statuses:

        overall_status = "Poor"

    elif "Acceptable" in statuses:

        overall_status = "Acceptable"

    else:

        overall_status = "Good"


    # ========================================================
    # QUALITY SCORE
    # ========================================================

    status_points = {
        "Good": 2,
        "Acceptable": 1,
        "Poor": 0
    }


    total_points = sum(
        status_points[
            status
        ]
        for status in statuses
    )


    quality_score = (
        total_points / 8
    ) * 100


    return {

        "overall_status":
            overall_status,

        "quality_score":
            round(
                quality_score,
                1
            ),

        "blur_score":
            round(
                float(blur_score),
                2
            ),

        "blur_status":
            blur_status,

        "brightness_score":
            round(
                brightness_score,
                2
            ),

        "brightness_status":
            brightness_status,

        "contrast_score":
            round(
                contrast_score,
                2
            ),

        "contrast_status":
            contrast_status,

        "resolution":
            f"{original_width} x {original_height}",

        "resolution_status":
            resolution_status
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        image_path = sys.argv[1]

    else:

        image_path = (
            "Dataset/FinalClean/Testing/"
            "glioma/Te-gl_203.jpg"
        )


    if not os.path.exists(
        image_path
    ):

        print(
            "Image not found:"
        )

        print(
            image_path
        )

        sys.exit(1)


    print("\n==========================================")
    print("NEUROVISION AI")
    print("MRI QUALITY ASSESSMENT")
    print("==========================================")

    print(
        "\nImage:",
        image_path
    )


    try:

        result = assess_mri_quality(
            image_path
        )

    except Exception as error:

        print(
            "\nERROR:",
            error
        )

        sys.exit(1)


    print("\n------------------------------------------")
    print("QUALITY RESULT")
    print("------------------------------------------")

    print(
        f'Overall Quality : '
        f'{result["overall_status"]}'
    )

    print(
        f'Quality Score   : '
        f'{result["quality_score"]}%'
    )

    print()

    print(
        f'Sharpness       : '
        f'{result["blur_score"]} '
        f'[{result["blur_status"]}]'
    )

    print(
        f'Brightness      : '
        f'{result["brightness_score"]} '
        f'[{result["brightness_status"]}]'
    )

    print(
        f'Contrast        : '
        f'{result["contrast_score"]} '
        f'[{result["contrast_status"]}]'
    )

    print(
        f'Resolution      : '
        f'{result["resolution"]} '
        f'[{result["resolution_status"]}]'
    )

    print("\n==========================================")
    print("QUALITY ASSESSMENT COMPLETED")
    print("==========================================")
