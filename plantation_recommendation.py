def get_plantation_recommendation(dist):
    """
    Generates plantation recommendations based on tree distribution
    from the uploaded image.
    """

    # ---- Safe extraction with defaults ----
    left = dist.get("left", 0)
    center = dist.get("center", 0)
    right = dist.get("right", 0)

    top = dist.get("top", 0)
    middle = dist.get("middle", 0)
    bottom = dist.get("bottom", 0)

    # ---- Horizontal & Vertical grouping ----
    horizontal = {
        "left": left,
        "center": center,
        "right": right
    }

    vertical = {
        "top": top,
        "middle": middle,
        "bottom": bottom
    }

    # ---- Find lowest density regions ----
    min_horizontal = min(horizontal, key=horizontal.get)
    min_vertical = min(vertical, key=vertical.get)

    # ---- Recommendation messages ----
    messages = []

    messages.append(
        f"The {min_vertical}–{min_horizontal} region has the lowest tree density "
        f"based on the uploaded image analysis."
    )

    messages.append(
        f"It is recommended to prioritize new tree plantation in the "
        f"{min_vertical}–{min_horizontal} region to improve green cover balance."
    )

    avg_density = sum(dist.values()) / len(dist)

    if left < avg_density and right < avg_density:
        messages.append(
            "Horizontal plantation is advised to balance vegetation across left and right regions."
        )

    if top < avg_density and bottom < avg_density:
        messages.append(
            "Vertical plantation is advised to enhance green cover in upper and lower regions."
        )

    if len(messages) == 2:
        messages.append(
            "Tree distribution appears relatively balanced. Focus on conserving existing mature trees."
        )

    # ✅ RETURN STRUCTURE EXPECTED BY HTML
    return {
        "horizontal": min_horizontal.capitalize(),
        "vertical": min_vertical.capitalize(),
        "message": " ".join(messages)
    }
