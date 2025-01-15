from fastapi import FastAPI, File, UploadFile, HTTPException
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text
from PIL import Image
from sentence_transformers import SentenceTransformer
import io
import requests
import base64
import numpy as np
from scipy.spatial.distance import cosine
import os

# Database URL from environment variable
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5445/postgres")

# FastAPI app
app = FastAPI()

# Database connection
database = Database(DATABASE_URL)
metadata = MetaData()

# Define the products table
products = Table(
    "products",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(50), nullable=False),
    Column("title", String(100), nullable=False),
    Column("cover_image_url", Text, nullable=True),
    Column("embeded", Text, nullable=True),
)

# Initialize the database engine
engine = create_engine(DATABASE_URL)

model = SentenceTransformer('clip-ViT-B-32')

# Function to create the products table if it doesn't exist
async def create_products_table():
    metadata.create_all(engine)

# Function to seed the products table
async def seed_products():
    fake_products = [
        {
            "name": "iPhone 15 Pro",
            "title": "Apple iPhone 15 Pro Max 256GB",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS2LMI6reRQmOHJaQkXyUVVtpWohVRjdSj2eQ&s",
        },
        {
            "name": "MacBook Pro",
            "title": "Apple MacBook Pro 16-inch M3 Max",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-2IE9TZRH9mP1wbkb2ugaY0mtk1PO3yHkwQ&s",
        },
        {
            "name": "Samsung S24",
            "title": "Samsung Galaxy S24 Ultra 512GB",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQX-C1rCcHFfBVQ9NnHSus-KojKc3nhYInSQ&s",
        },
        {
            "name": "iPad Pro",
            "title": "Apple iPad Pro 12.9-inch M2",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2pKMV_UxIlCsS3NDIbqzCsCm_Mvgz41mtOg&s",
        },
        {
            "name": "Dell XPS",
            "title": "Dell XPS 15 OLED Touch",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7Yx5RswE603ioLodzFRpx8b8WABHSRvgJTg&s",
        },
        {
            "name": "Sony WH-1000XM5",
            "title": "Sony WH-1000XM5 Wireless Headphones",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZopYAXkhG4jwmrVayByfeI3W1B7eG4Wjqjg&s",
        },
        {
            "name": "Nintendo Switch",
            "title": "Nintendo Switch OLED Model",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZopYAXkhG4jwmrVayByfeI3W1B7eG4Wjqjg&s",
        },
        {
            "name": "PS5",
            "title": "PlayStation 5 Digital Edition",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSru-M_eY5CoNFLjezhdy-UHT9V_qoFSWNdfw&s",
        },
        {
            "name": "Xbox Series X",
            "title": "Microsoft Xbox Series X 1TB",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIh6f_0wfSiAJNLWAXuaKsW_0VjL1UUBgE6w&s",
        },
        {
            "name": "AirPods Pro",
            "title": "Apple AirPods Pro 2nd Generation",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQs6sUI4fUp1lqBZ7Eu0QbAsDAjAfJARDFE5A&s",
        },
        {
            "name": "Galaxy Buds",
            "title": "Samsung Galaxy Buds2 Pro",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRc4AsLCNGeP6CA04qmWT4SFtCq-QBoGCdciw&s",
        },
        {
            "name": "Surface Laptop",
            "title": "Microsoft Surface Laptop 5",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQyYJIQelWB8X58MRWB498NmIZthnjCY2VFZA&s",
        },
        {
            "name": "Apple Watch",
            "title": "Apple Watch Series 9 GPS + Cellular",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHP2LZGSzIyG_sOCS-GiMKQL-Jo-oGXOI5fQ&s",
        },
        {
            "name": "Galaxy Watch",
            "title": "Samsung Galaxy Watch 6 Classic",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTL1VDbblNuNxgSvFUfMlW6hXJocm6ToK1KlA&s",
        },
        {
            "name": "Canon EOS",
            "title": "Canon EOS R5 Mirrorless Camera",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGkrNgNlKHsDsFeBwJATG8KD1lo8Qw7sg73w&s",
        },
        {
            "name": "DJI Drone",
            "title": "DJI Air 3 Fly More Combo",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRuDfXP2pHpWBIWucwZLWz5OXlEeSpvHC8lDA&s",
        },
        {
            "name": "Kindle",
            "title": "Amazon Kindle Paperwhite 16GB",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLdG4BPJa2CWfa4oo4bZO5BgTgiIifozuZSw&s",
        },
        {
            "name": "GoPro",
            "title": "GoPro HERO12 Black",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHaaTB_NE7NOzeAzZi4GXhnfG0rbgxQXPA6A&s",
        },
        {
            "name": "Bose QC",
            "title": "Bose QuietComfort Ultra Headphones",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcShgRzDB5KJCknGkrf0SIUvvloVSHEV8hVCow&s",
        },
        {
            "name": "LG OLED",
            "title": "LG C3 65-inch 4K OLED TV",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHJAt5oUK1rb5RlS7mtCDeZE2smKWn9K1FWA&s",
        },
        {
            "name": "Sonos",
            "title": "Sonos Arc Soundbar",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvz1D__ZetRTgxNW1iJprJUi6MuEUlvBPc9Q&s",
        },
        {
            "name": "Dyson V15",
            "title": "Dyson V15 Detect Absolute",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSy27zko9gkpeP5ryprFh2lXxN34tBOfHPuMA&s",
        },
        {
            "name": "Ring Doorbell",
            "title": "Ring Video Doorbell Pro 2",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSr4DKnWqQHVo08pZWpKL8ditR0ja-kIe78TQ&s",
        },
        {
            "name": "Nest Thermostat",
            "title": "Google Nest Learning Thermostat",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYxy5o-a6aLUu73nH94_DjPXhCjDTEtPKfLQ&s",
        },
        {
            "name": "Philips Hue",
            "title": "Philips Hue Starter Kit",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTCruXkRgofWAu1SbPqcKx9imcmvEKqCoCdbQ&s",
        },
        {
            "name": "Nvidia RTX",
            "title": "Nvidia GeForce RTX 4090",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTu4nFfEDfHxk7bwTa3HC-owzquoXfBfmUlg&s",
        },
        {
            "name": "AMD Ryzen",
            "title": "AMD Ryzen 9 7950X",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLq2oN0u9fM1PBRHLSWVwDQ2az65k_0vyzug&s",
        },
        {
            "name": "Steam Deck",
            "title": "Steam Deck 512GB",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAFnKdywjCSbXzM7f50aqFPb1tqUamu4Ti4A&s",
        },
        {
            "name": "ROG Laptop",
            "title": "ASUS ROG Zephyrus G14",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSqE53rcJyOnhFrIoo3I2jvG8yjUKizP_aCg&s",
        },
        {
            "name": "Apple TV",
            "title": "Apple TV 4K 128GB",
            "cover_image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZR61OWUaEPEZOh8xxzMkZaPasLcuHTaN0PA&s",
        },
    ]

    for product in fake_products:
        image_response = requests.get(product["cover_image_url"])
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        embedding = model.encode(image).tobytes()
        embedding_str = base64.b64encode(embedding).decode("utf-8")  # Encode to Base64 string
        
        query = products.insert().values(
            name=product["name"],
            title=product["title"],
            cover_image_url=product["cover_image_url"],
            embeded=embedding_str,
        )
        await database.execute(query)


    print("Product seeding complete")

# Startup event to initialize and seed the database
@app.on_event("startup")
async def startup():
    await database.connect()
    # await create_products_table()
    
    # Clear existing data in the products table
    # await database.execute(products.delete())
    
    # Seed products data
    # await seed_products()

# Shutdown event to disconnect the database
@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/")
async def root():
    return {"message": "Welcome to the Product API"}
    
# Function to get the embedding from an image
def get_image_embedding(image: Image.Image) -> np.ndarray:
    # Convert the image to the appropriate format and generate the embedding
    embedding = model.encode(image)
    return embedding

# Helper function to calculate cosine similarity
def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return 1 - cosine(embedding1, embedding2)

# New API endpoint to upload an image and get similar products
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Read image data
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Get the embedding for the uploaded image
    uploaded_image_embedding = get_image_embedding(image)

    # Fetch all products from the database
    query = products.select()
    existing_products = await database.fetch_all(query)

    # Calculate similarity for each product and store the results
    similarities = []
    for product in existing_products:
        # Decode the stored embedding from Base64
        product_embedding = base64.b64decode(product["embeded"])
        product_embedding = np.frombuffer(product_embedding, dtype=np.float32)
        
        # Calculate the cosine similarity
        similarity = calculate_similarity(uploaded_image_embedding, product_embedding)
        
        # Convert similarity to a native float
        similarity = float(similarity)
        
        if similarity > 0.6:
            similarities.append((product, similarity))

    # Sort products by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the sorted list of products with similarity scores
    result = [
        {
            "name": product["name"],
            "title": product["title"],
            "cover_image_url": product["cover_image_url"],
            "similarity": similarity,
        }
        for product, similarity in similarities
    ]
    
    return result