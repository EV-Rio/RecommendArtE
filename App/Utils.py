import timm
import io
from PIL import Image
import numpy as np
import sqlite3
import torch
import streamlit as st
from sklearn.neighbors import KDTree

# Devuelve un cursor a artworks.db
def get_db_connection():
    db_path = r"..\artworks.db"
    conn = sqlite3.connect(db_path)
    db_cursor = conn.cursor()
    return db_cursor

# Devuelve un cursor a artworks_raw.db
def get_db_connection_raw():
    db_path = r"..\artworks_raw.db"
    conn = sqlite3.connect(db_path)
    db_cursor = conn.cursor()
    return db_cursor

# Inicializa el modelo ResNet50
@st.cache_data
def initialize_model():
    resnet50_model = timm.create_model('resnet50',pretrained=True,num_classes=0)
    return resnet50_model


# Crea pipeline de preprocesado para un modelo concreto
from timm.data import create_transform
def create_pipeline(model):
    config = model.default_cfg
    transform = create_transform(
        input_size=config['input_size'],
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        crop_pct=config['crop_pct']
    )
    return transform


# Devuelve el vector de un blob
def get_embedding(blob, model, transform):
    img = Image.open(io.BytesIO(blob)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding_raw = model(img_tensor)
    embedding = embedding_raw.detach().cpu().squeeze().numpy()
    return embedding

# Devuelve todos los vectores de artworks.db
@st.cache_data
def get_db_embeddings(_db_cursor):
	_db_cursor.execute("SELECT painting, image_embedding FROM artworks")
	embeddings_list = _db_cursor.fetchall()
	return embeddings_list

# Devuelve el vector de una pintura concreta de la base de datos
def get_blob(painting_title, db_cursor):
    db_cursor.execute("SELECT image_blob FROM artworks WHERE painting = ? LIMIT 1",
                      (painting_title,)
                    )
    new_blob = db_cursor.fetchone()[0]
    return new_blob

# Devuelve el creador, el museo y el movimiento artístico de una pintura concreta de la base de datos
def get_painting_creator_museum_movement(painting_title, db_cursor):
    db_cursor.execute("SELECT creator, museum, movement FROM artworks WHERE painting = ?", (painting_title,))
    creator, museum, movement = db_cursor.fetchone()
    return creator, museum, movement


# Obtención del KDTree 
@st.cache_data
def build_kdtree(_db_cursor):
    # Obtención de los títulos y vectores de la base de datos
    _db_cursor.execute("SELECT painting, image_embedding FROM artworks")
    embeddings_list = _db_cursor.fetchall()
    # Separamos nombres y vectores
    painting_names = [item[0] for item in embeddings_list]
    # Los vectores están guardados en formato bytes en la base de datos, conversión a float
    embeddings_array = np.array([
        np.frombuffer(item[1], dtype=np.float32) 
        for item in embeddings_list
    ])
    # Normalizar vectores 
    embeddings_normalized = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    # Construir KDTree con vectores normalizados
    tree = KDTree(embeddings_normalized, leaf_size=40)
    return tree, painting_names

# Devuelve los títulos de las k pinturas más cercanas
def get_closest_paintings_tree(new_embedding, tree, painting_names, k=3):
    # Normalizar el nuevo vector a la misma escala que los demás
    new_embedding_normalized = new_embedding / np.linalg.norm(new_embedding)
    # Consultar KDTree por los vecinos más cercanos
    distances, indices = tree.query([new_embedding_normalized], k=k)
    closest_paintings = [painting_names[i] for i in indices[0]]
    cosine_distances = distances[0]
    return closest_paintings

# Obtener el link a la ficha de Wikidata
def get_wikidata_ref(painting, db_cursor, db_cursor_raw):
    db_cursor.execute("SELECT id FROM artworks WHERE painting = ?", (painting,))
    image_id = db_cursor.fetchone()[0]
    db_cursor_raw.execute("SELECT painting FROM artworks_raw WHERE id = ? ", (image_id,))
    painting_url = db_cursor_raw.fetchone()
    return painting_url