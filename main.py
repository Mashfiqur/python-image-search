from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import databases
import sqlalchemy
import os
from typing import List
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import io

# Initialize FastAPI app
app = FastAPI()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5445/postgres")
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Define Products table
products = sqlalchemy.Table(
    "products",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("title", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("cover_image_url", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("embedding", sqlalchemy.LargeBinary, nullable=False),  # Store embeddings
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

# Load a pre-trained image embedding model
model = SentenceTransformer('clip-ViT-B-32')

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Seeder function to populate products
async def seed_products():
    fake_products = [
        {
            "name": "iPhone",
            "title": "Apple iPhone 13",
            "cover_image_url": "https://example.com/iphone.jpg",
        },
        {
            "name": "Samsung Galaxy",
            "title": "Samsung Galaxy S21",
            "cover_image_url": "https://example.com/galaxy.jpg",
        },
        {
            "name": "MacBook",
            "title": "Apple MacBook Air",
            "cover_image_url": "https://example.com/macbook.jpg",
        },
    ]

    for product in fake_products:
        # Download and process the cover image
        image_response = requests.get(product["cover_image_url"])
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        embedding = model.encode(image).tobytes()

        # Insert product into the database
        query = products.insert().values(
            name=product["name"],
            title=product["title"],
            cover_image_url=product["cover_image_url"],
            embedding=embedding,
        )
        await database.execute(query)

# Call seeding function
@app.on_event("startup")
async def seed_on_startup():
    await seed_products()

# Endpoint to search products based on an uploaded image
@app.post("/search-products/")
async def search_products(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Process the uploaded image
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    query_embedding = model.encode(image)

    # Fetch all products and compute similarity
    query = products.select()
    all_products = await database.fetch_all(query)

    # Compute similarity scores
    results = []
    for product in all_products:
        product_embedding = np.frombuffer(product["embedding"], dtype=np.float32)
        similarity = np.dot(query_embedding, product_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding))
        results.append({**product, "similarity": similarity})

    # Sort results by similarity score in descending order
    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return {"results": sorted_results}
