from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
import os
import cv2
from urllib.parse import quote
from aqi_model import analyze_environment

# -------------------------------
# Import project modules
# -------------------------------
from yolo_tree_detection import detect_trees_yolo
from tree_heatmap import create_tree_heatmap
from preprocessing_pipeline import preprocessing_pipeline
from green_cover import green_cover_estimation
from optimal_path import generate_optimal_path
from tree_info import TREE_INFO
from tree_distribution import calculate_tree_distribution
from plantation_recommendation import get_plantation_recommendation

# -------------------------------
# Flask app setup
# -------------------------------
app = Flask(__name__)
app.secret_key = "greenpath_secret_key"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/about")
def about():
    return render_template("about.html")


# =====================================================
# HOME PAGE – TREE DETECTION
# =====================================================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            session["current_image"] = file.filename

            tree_count, boxes = detect_trees_yolo(image_path, OUTPUT_FOLDER)

            image = cv2.imread(image_path)
            if image is not None:
                heatmap_img = create_tree_heatmap(image, boxes)
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, "tree_heatmap.png"), heatmap_img)

            preprocessing_pipeline(image_path, OUTPUT_FOLDER)
            session["tree_count"] = tree_count

        return redirect(url_for("home"))

    return render_template(
        "index.html",
        tree_count=session.get("tree_count"),
        uploaded_image=session.get("current_image")
    )


# =====================================================
# 🌫 AQI ESTIMATION (Image-Based)
# =====================================================
@app.route("/aqi")
def aqi():
    current_image = session.get("current_image")

    if not current_image:
        return redirect(url_for("home"))

    image_path = os.path.join(UPLOAD_FOLDER, current_image)

    if not os.path.exists(image_path):
        return redirect(url_for("home"))

    # 🔥 Run AQI Model
    aqi_result = analyze_environment(image_path)

    return render_template(
        "aqi.html",
        aqi_result=aqi_result,
        uploaded_image=current_image
    )


# =====================================================
# 🌳 TREE AWARENESS
# =====================================================
@app.route("/tree-awareness")
def tree_awareness():
    trees_with_images = []

    for tree in TREE_INFO:
        tree_copy = tree.copy()
        tree_name = tree.get("name", "Tree")
        query = quote(tree_name + " tree")
        tree_copy["image_url"] = (
            f"https://commons.wikimedia.org/wiki/Special:FilePath/{query}.jpg"
        )
        trees_with_images.append(tree_copy)

    return render_template("tree_awareness.html", trees=trees_with_images)


# =====================================================
# 🌍 TREE DISTRIBUTION
# =====================================================
@app.route("/tree-distribution")
def tree_distribution():
    if "current_image" not in session:
        return "Upload an image first"

    image_path = os.path.join(UPLOAD_FOLDER, session["current_image"])
    _, boxes = detect_trees_yolo(image_path, OUTPUT_FOLDER)

    dist = calculate_tree_distribution(image_path, boxes)
    session["tree_distribution"] = dist

    return render_template("tree_distribution.html", dist=dist)


# =====================================================
# 🌱 PLANTATION RECOMMENDATION
# =====================================================
@app.route("/plantation")
def plantation():
    if "current_image" not in session:
        return "Upload an image first"

    image_path = os.path.join(UPLOAD_FOLDER, session["current_image"])

    _, boxes = detect_trees_yolo(image_path, OUTPUT_FOLDER)
    dist = calculate_tree_distribution(image_path, boxes)

    result = get_plantation_recommendation(dist)

    return render_template("plantation.html", result=result)


# =====================================================
# OPTIMAL PATH
# =====================================================
@app.route("/optimal-path")
def optimal_path():
    if "current_image" not in session:
        return "Upload and analyze an image first"

    input_image = os.path.join(UPLOAD_FOLDER, session["current_image"])
    name, _ = os.path.splitext(session["current_image"])
    output_filename = f"optimal_{name}.png"
    output_image = os.path.join(OUTPUT_FOLDER, output_filename)

    generate_optimal_path(input_image, output_image)

    return render_template("optimal_path.html", output_image=output_filename)


# =====================================================
# HEATMAP / PIPELINE
# =====================================================
@app.route("/heatmap")
def heatmap():
    if "current_image" not in session:
        return "Upload and analyze an image first"
    return render_template("heatmap.html")


@app.route("/pipeline")
def pipeline():
    if "current_image" not in session:
        return "Upload and analyze an image first"
    return render_template("pipeline.html")

# =====================================================
# 🌱 GREEN COVER ESTIMATOR
# =====================================================
@app.route("/green-cover")
def green_cover():
    if "current_image" not in session:
        return "Upload and analyze an image first"

    image_path = os.path.join(UPLOAD_FOLDER, session["current_image"])

    green, non_green, green_img, non_green_img = green_cover_estimation(
        image_path, OUTPUT_FOLDER
    )

    # ✅ ADD AQI (no logic change)
    aqi_result = analyze_environment(image_path)

    return render_template(
        "green_cover.html",
        green=green,
        non_green=non_green,
        green_img=green_img,
        non_green_img=non_green_img,
        aqi_result=aqi_result   # added
    )
@app.route("/carbon")
def carbon():
    if "current_image" not in session:
        return redirect(url_for("home"))

    image_path = os.path.join(UPLOAD_FOLDER, session["current_image"])

    tree_count, _ = detect_trees_yolo(image_path, OUTPUT_FOLDER)

    co2_absorption = tree_count * 22

    return render_template(
        "carbon_absorption.html",
        co2_absorption=co2_absorption
    )
@app.route("/suggestions")
def suggestions():
    return redirect(url_for("environmental_report"))



@app.route("/environmental-report")
def environmental_report():

    if "current_image" not in session:
        return redirect(url_for("home"))

    image_path = os.path.join(UPLOAD_FOLDER, session["current_image"])

    # -------------------------------
    # TREE DETECTION
    # -------------------------------
    tree_count, boxes = detect_trees_yolo(image_path, OUTPUT_FOLDER)
    distribution = calculate_tree_distribution(image_path, boxes)

    # -------------------------------
    # GREEN COVER
    # -------------------------------
    green, non_green, green_img, non_green_img = green_cover_estimation(
        image_path, OUTPUT_FOLDER
    )

    # -------------------------------
    # AQI
    # -------------------------------
    aqi_result = analyze_environment(image_path)

    # -------------------------------
    # CARBON
    # -------------------------------
    co2_absorption = tree_count * 22

    # -------------------------------
    # ECO INDEX
    # -------------------------------
    eco_index = int((green * 0.4) + (tree_count * 0.3))
    eco_status = "Healthy 🌿" if eco_index > 50 else "Needs Improvement ⚠"

    # -------------------------------
    # HEAT REDUCTION
    # -------------------------------
    BASE_TEMP = 35
    heat_reduction = round((tree_count * 0.02) + (green * 0.03), 2)
    final_temp = round(BASE_TEMP - heat_reduction, 2)

    if heat_reduction < 1:
        cooling_level = "Low Cooling Effect 🟡"
    elif heat_reduction < 3:
        cooling_level = "Moderate Cooling Effect 🟢"
    else:
        cooling_level = "High Cooling Effect 🔵"

    # -------------------------------
    # SMART ZONE (IMPORTANT — INSIDE FUNCTION)
    # -------------------------------
    zones = {}

    for zone, count in distribution.items():
        zone_score = (count * 3) + (green * 0.5)

        if zone_score > 70:
            zones[zone] = "Highly Green Zone 🌳"
        elif zone_score > 40:
            zones[zone] = "Moderate Zone 🌿"
        elif zone_score > 20:
            zones[zone] = "Heat Risk Zone 🔥"
        else:
            zones[zone] = "Critical Zone ⚠"

    # -------------------------------
    # RETURN
    # -------------------------------
    return render_template(
        "environmental_report.html",
        tree_count=tree_count,
        distribution=distribution,
        green=green,
        non_green=non_green,
        green_img=green_img,
        non_green_img=non_green_img,
        aqi_result=aqi_result,
        co2_absorption=co2_absorption,
        eco_index=eco_index,
        eco_status=eco_status,
        heat_reduction=heat_reduction,
        final_temp=final_temp,
        cooling_level=cooling_level,
        zones=zones
    )


# =====================================================
# 🌡 URBAN HEAT REDUCTION
# =====================================================
@app.route('/urban-heat')
def urban_heat():

    if "current_image" not in session:
        return redirect(url_for("home"))

    image_path = os.path.join(UPLOAD_FOLDER, session["current_image"])

    tree_count, _ = detect_trees_yolo(image_path, OUTPUT_FOLDER)
    green, _, _, _ = green_cover_estimation(image_path, OUTPUT_FOLDER)

    BASE_TEMP = 35

    reduction = round((tree_count * 0.02) + (green * 0.03), 2)
    final_temp = round(BASE_TEMP - reduction, 2)

    # ✅ ADD COOLING LEVEL HERE
    if reduction < 1:
        cooling_level = "Low Cooling Effect 🟡"
    elif reduction < 3:
        cooling_level = "Moderate Cooling Effect 🟢"
    else:
        cooling_level = "High Cooling Effect 🔵"

    return render_template(
        "urban_heat.html",
        tree_count=tree_count,
        green=green,
        reduction=reduction,
        final_temp=final_temp,
        cooling_level=cooling_level   # 👈 PASS IT
    )

# =====================================================
# 🌳 TREE IMPACT SIMULATOR
# =====================================================
@app.route("/simulate", methods=["GET", "POST"])
def simulate():

    if "current_image" not in session:
        return redirect(url_for("home"))

    image_path = os.path.join(UPLOAD_FOLDER, session["current_image"])

    tree_count, _ = detect_trees_yolo(image_path, OUTPUT_FOLDER)
    green, _, _, _ = green_cover_estimation(image_path, OUTPUT_FOLDER)

    BASE_TEMP = 35
    current_reduction = (tree_count * 0.02) + (green * 0.03)
    current_temp = round(BASE_TEMP - current_reduction, 2)

    future_temp = None
    improvement = None
    simulated_filename = None

    if request.method == "POST":
        try:
            extra_trees = max(0, int(request.form.get("extra_trees", 0)))
        except:
            extra_trees = 0

        # ----------------------------
        # TEMPERATURE CALCULATION
        # ----------------------------
        future_tree_count = tree_count + extra_trees
        future_green = min(green + (extra_trees * 0.2), 100)

        future_reduction = (future_tree_count * 0.02) + (future_green * 0.03)
        future_temp = round(BASE_TEMP - future_reduction, 2)
        improvement = round(current_temp - future_temp, 2)

        # ----------------------------
        # IMAGE SIMULATION
        # ----------------------------
        img = cv2.imread(image_path)

        if img is not None:
            h, w, _ = img.shape

            for _ in range(extra_trees):
                x = random.randint(30, w - 30)
                y = random.randint(30, h - 30)

                # Draw green circle (tree canopy)
                cv2.circle(img, (x, y), 15, (0, 150, 0), -1)

                # Draw trunk
                cv2.rectangle(img, (x-3, y+15), (x+3, y+25), (42, 42, 165), -1)

            simulated_filename = "simulated_" + session["current_image"]
            output_path = os.path.join(OUTPUT_FOLDER, simulated_filename)
            cv2.imwrite(output_path, img)

    return render_template(
        "simulate.html",
        tree_count=tree_count,
        green=green,
        current_temp=current_temp,
        future_temp=future_temp,
        improvement=improvement,
        simulated_image=simulated_filename
    )


# =====================================================
# 🌍 SMART ZONE CLASSIFICATION
# =====================================================
@app.route("/smart-zone")
def smart_zone():

    if "current_image" not in session:
        return redirect(url_for("home"))

    image_path = os.path.join(UPLOAD_FOLDER, session["current_image"])

    # Detect trees
    tree_count, boxes = detect_trees_yolo(image_path, OUTPUT_FOLDER)
    distribution = calculate_tree_distribution(image_path, boxes)

    # Get green cover
    green, _, _, _ = green_cover_estimation(image_path, OUTPUT_FOLDER)

    # Smart Zone Logic
    zones = {}

    for zone, count in distribution.items():

        zone_score = (count * 3) + (green * 0.5)

        if zone_score > 70:
            zones[zone] = "Highly Green Zone 🌳"
        elif zone_score > 40:
            zones[zone] = "Moderate Zone 🌿"
        elif zone_score > 20:
            zones[zone] = "Heat Risk Zone 🔥"
        else:
            zones[zone] = "Critical Zone ⚠"

    # ✅ IMPORTANT: RETURN TEMPLATE
    return render_template(
        "smart_zone.html",
        zones=zones,
        tree_count=tree_count,
        green=green
    )

# =====================================================
# STATIC FILES
# =====================================================
@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


# =====================================================
# Disable cache
# =====================================================
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response


# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
