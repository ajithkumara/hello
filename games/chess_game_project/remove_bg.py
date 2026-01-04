from PIL import Image

def remove_white_bg(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        # Check if pixel is close to white (allow some variance)
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0)) # Transparent
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(output_path, "PNG")
    print(f"Saved transparent image to {output_path}")

import sys
if __name__ == "__main__":
    if len(sys.argv) > 2:
        remove_white_bg(sys.argv[1], sys.argv[2])
    else:
        # Default behavior for testing
        try:
             remove_white_bg("car_texture_3d.png", "car_texture_3d.png")
        except:
             pass
