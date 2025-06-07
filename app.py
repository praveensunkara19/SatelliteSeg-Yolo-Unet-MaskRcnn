import gradio as gr
from PIL import Image
from utils.predictor import predict_image, predict_map
import folium
from folium.plugins import MousePosition
from folium.raster_layers import TileLayer

def create_folium_map(lat, lon, zoom):
    fmap = folium.Map(location=[lat, lon], zoom_start=zoom)

    TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        overlay=False,
        control=True
    ).add_to(fmap)

    marker = folium.Marker(location=[lat, lon], draggable=True)
    marker.add_to(fmap)

    MousePosition(
        position='topright',
        separator=' | ',
        prefix='Lat | Lon:',
        num_digits=5
    ).add_to(fmap)

    # JavaScript to update textbox values and trigger Gradio update
    fmap.get_root().html.add_child(folium.Element(f"""
<script>
function triggerInputChange(input, value) {{
    input.value = value;
    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
}}

function handleDragEnd(e) {{
    var lat = e.target.getLatLng().lat.toFixed(5);
    var lng = e.target.getLatLng().lng.toFixed(5);

    let latInput = document.querySelector('input#lat_input');
    let lonInput = document.querySelector('input#lon_input');

    if (latInput && lonInput) {{
        triggerInputChange(latInput, lat);
        triggerInputChange(lonInput, lng);
    }}
}}

function handleZoomChange(map) {{
    let zoomInput = document.querySelector('input#zoom_input');
    if (zoomInput) {{
        triggerInputChange(zoomInput, map.getZoom());
    }}
}}

setTimeout(() => {{
    var leafletMap = document.querySelector('.leaflet-container')._leaflet_map;

    leafletMap.eachLayer(layer => {{
        if (layer instanceof L.Marker && layer.options.draggable) {{
            layer.on('dragend', handleDragEnd);
        }}
    }});

    leafletMap.on('zoomend', () => handleZoomChange(leafletMap));
}}, 500);
</script>
"""))


    return fmap._repr_html_()


def predict_uploaded_image(model_type, image):
    result = predict_image(image, model_type)
    if "error" in result:
        raise gr.Error(f"Image prediction failed: {result['error']}")
    return (
        [result["original"], result["overlay"]],
        result["split_images"],
        result["areas"]
    )


def capture_map_and_predict(lat, lon, zoom, model_type):
    try:
        result = predict_map(model_type, (lat, lon), zoom)
        if "error" in result:
            raise gr.Error(f"Map prediction failed: {result['error']}")
        return (
            [result["original"], result["overlay"]],
            result["split_images"],
            result["areas"]
        )
    except Exception as e:
        raise gr.Error(f"Map prediction failed: {str(e)}")


def load_and_predict_test_image(model_type, image_path):
    image = Image.open(image_path).convert("RGB")
    return predict_uploaded_image(model_type, image)


with gr.Blocks() as demo:
    gr.Markdown("# 🛰️ Satellite Image Segmentation with Folium Map (Esri Satellite Tiles)")

    with gr.Row():
        input_type = gr.Radio(["Upload Image", "Capture from Map"], value="Upload Image", label="Input Method")
        model_type = gr.Radio(["YOLOv11", "UNet", "MaskRCNN"], value="YOLOv11", label="Select Segmentation Model")

    image_input = gr.Image(label="Upload Image", type="pil", visible=True)

    with gr.Row(visible=True) as test_image_row:
        gr.Markdown("### 🧪 Quick Test with Sample Images")

    with gr.Row(visible=True) as test_gallery:
        test1 = gr.Image(value="assets/test_images/f1.jpg", label="Test Image 1", interactive=True)
        test2 = gr.Image(value="assets/test_images/f2.jpg", label="Test Image 2", interactive=True)
        test3 = gr.Image(value="assets/test_images/f3.jpg", label="Test Image 3", interactive=True)

    with gr.Column(visible=False) as map_controls:
        gr.Markdown("📍 **Drag the marker on the map or move the map of your intrest. Enter the coordinate in the lon|lat section then click predict.**")

        lat_input = gr.Textbox(value="22.9749", label="Latitude", elem_id="lat_input")
        lon_input = gr.Textbox(value="76.2168", label="Longitude", elem_id="lon_input")
        zoom_input = gr.Slider(14, 20, value=16, step=1, label="Zoom Level", elem_id="zoom_input")


        map_html = gr.HTML(value=create_folium_map(22.9749, 76.2168, 16))
        refresh_map_btn = gr.Button("🔄 Refresh Map View")
        map_predict_btn = gr.Button("📸 Capture Map and Predict")

    predict_btn = gr.Button("🔍 Predict", visible=True)

    with gr.Group():
        gr.Markdown("### 🖼️ Original vs Predicted Image")
        orig_pred_gallery = gr.Gallery(label="Original and Prediction", columns=2, rows=1)

    with gr.Group():
        gr.Markdown("### 📊 Class-wise Segmented Outputs")
        classwise_gallery = gr.Gallery(label="Per-Class Segmentation", columns=3, rows=2)

    area_json = gr.JSON(label="📐 Class-wise Area Breakdown")

    def toggle_input(choice):
        show_upload = choice == "Upload Image"
        return {
            image_input: gr.update(visible=show_upload),
            map_controls: gr.update(visible=not show_upload),
            predict_btn: gr.update(visible=show_upload),
            map_predict_btn: gr.update(visible=not show_upload),
            test_image_row: gr.update(visible=show_upload),
            test_gallery: gr.update(visible=show_upload)
        }

    input_type.change(
        toggle_input,
        input_type,
        [image_input, map_controls, predict_btn, map_predict_btn, test_image_row, test_gallery]
    )

    predict_btn.click(
        predict_uploaded_image,
        [model_type, image_input],
        [orig_pred_gallery, classwise_gallery, area_json]
    )

    refresh_map_btn.click(
        lambda lat, lon, zoom: create_folium_map(float(lat), float(lon), zoom),
        [lat_input, lon_input, zoom_input],
        map_html
    )

    map_predict_btn.click(
        lambda lat, lon, zoom, model_type: capture_map_and_predict(float(lat), float(lon), zoom, model_type),
        [lat_input, lon_input, zoom_input, model_type],
        [orig_pred_gallery, classwise_gallery, area_json]
    )

    test1.select(lambda model: load_and_predict_test_image(model, "assets/test_images/f1.jpg"), [model_type], [orig_pred_gallery, classwise_gallery, area_json])
    test2.select(lambda model: load_and_predict_test_image(model, "assets/test_images/f2.jpg"), [model_type], [orig_pred_gallery, classwise_gallery, area_json])
    test3.select(lambda model: load_and_predict_test_image(model, "assets/test_images/f3.jpg"), [model_type], [orig_pred_gallery, classwise_gallery, area_json])

if __name__ == "__main__":
    demo.launch()
