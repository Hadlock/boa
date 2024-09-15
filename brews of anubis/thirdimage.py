from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import random

def reduce_to_32768_colors(image):
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Reduce colors to 15-bit (5 bits per channel)
    image_np = (image_np // 8) * 8
    
    # Convert the numpy array back to an image
    new_image = Image.fromarray(image_np)
    return new_image

def reduce_to_54_colors(image):
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Reshape the image to be a list of pixels
    pixels = image_np.reshape(-1, 3)
    
    # Perform k-means clustering to reduce the number of colors
    kmeans = KMeans(n_clusters=54, random_state=0).fit(pixels)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshape the pixels to the original image shape
    new_image_np = new_colors.reshape(image_np.shape).astype(np.uint8)
    
    # Convert the numpy array back to an image
    new_image = Image.fromarray(new_image_np)
    return new_image

def resize_image(image, new_height=144):
    # Calculate the new width to maintain the aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    #new_width = 160 #int(new_height * aspect_ratio)
    
    # Resize the image to Nx144 using nearest neighbor interpolation
    resized_image = image.resize((new_width, new_height), Image.NEAREST)
    
    # Ensure the width is divisible by 8
    if resized_image.width % 8 != 0:
        new_width = (resized_image.width // 8) * 8
        resized_image = resized_image.crop((0, 0, new_width, resized_image.height))
    
    return resized_image

def get_tiles(image, tile_size=(8, 8)):
    tiles = []
    width, height = image.size
    for y in range(0, height, tile_size[1]):
        for x in range(0, width, tile_size[0]):
            tile = image.crop((x, y, x + tile_size[0], y + tile_size[1]))
            tiles.append(np.array(tile))
    return tiles

def randomly_select_tiles(tiles, num_tiles=384):
    if len(tiles) < num_tiles:
        print(f"Warning: Only {len(tiles)} tiles available, using all tiles.")
        return tiles
    return random.sample(tiles, num_tiles)

def find_closest_tile(tile, selected_tiles):
    min_distance = float('inf')
    closest_tile = None
    for selected_tile in selected_tiles:
        distance = np.sum((tile - selected_tile) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_tile = selected_tile
    return closest_tile

def reassemble_image_with_selected_tiles(tiles, selected_tiles, original_size, tile_size=(8, 8)):
    width, height = original_size
    new_image = Image.new('RGB', (width, height))
    
    tile_index = 0
    for y in range(0, height, tile_size[1]):
        for x in range(0, width, tile_size[0]):
            tile = tiles[tile_index]
            closest_tile = find_closest_tile(tile, selected_tiles)
            closest_tile_image = Image.fromarray(closest_tile)
            new_image.paste(closest_tile_image, (x, y))
            tile_index += 1
    
    return new_image

def create_sprite_sheet(unique_tiles, tile_size=(8, 8), columns=16):
    rows = (len(unique_tiles) + columns - 1) // columns
    sprite_sheet = Image.new('RGB', (columns * tile_size[0], rows * tile_size[1]))
    
    for index, tile in enumerate(unique_tiles):
        x = (index % columns) * tile_size[0]
        y = (index // columns) * tile_size[1]
        tile_image = Image.fromarray(tile)
        sprite_sheet.paste(tile_image, (x, y))
    
    return sprite_sheet

def reduce_colors(image_path, output_path='output_image.png', tile_output_path='tile_output.png', sprite_sheet_path='sprite_sheet.png', resized_output_path='simply_resized.png'):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Reduce to 32,768 colors
    image_32768 = reduce_to_32768_colors(image)
    
    # Further reduce to 54 colors
    image_54 = reduce_to_54_colors(image_32768)
    
    # Resize the image to Nx144
    resized_image = resize_image(image_54, new_height=144)
    
    # Save the resized image
    resized_image.save(resized_output_path)
    print(f"Resized image saved to {resized_output_path}")
    
    # Get the tiles
    tiles = get_tiles(resized_image)
    
    # Randomly select 384 tiles
    selected_tiles = randomly_select_tiles(tiles, num_tiles=384)
    
    # Reassemble the image using the selected tiles
    reassembled_image = reassemble_image_with_selected_tiles(tiles, selected_tiles, resized_image.size)
    
    # Save the reassembled image
    reassembled_image.save(tile_output_path)
    print(f"Image saved to {tile_output_path}")
    
    # Create and save the sprite sheet
    sprite_sheet = create_sprite_sheet(selected_tiles)
    sprite_sheet.save(sprite_sheet_path)
    print(f"Sprite sheet saved to {sprite_sheet_path}")

# Example usage
image_path = 'smallmoon1.jpg'
reduce_colors(image_path, output_path='reduced_colors_image.png', tile_output_path='tile_output.png', sprite_sheet_path='sprite_sheet.png', resized_output_path='simply_resized.png')
