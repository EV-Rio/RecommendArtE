import streamlit as st
import Utils
from PIL import Image, ImageOps
import io
import pandas as pd

# Título
st.markdown(
    '<h1 style="text-align: center;"><span style="font-weight: 300;">🖼</span> RecomendArtE <span style="font-weight: 10;">🖼</span></h1>',
    unsafe_allow_html=True
)

# Introducción al proyecto
st.text("RecomendArtE es un sistema de recomendación basado en cuadros" \
" de pintores españoles. Surgió a partir de una preocupación; el auge del" \
" arte hecho con ia y la consecuente desaparición de los artistas reales" \
" de algunos ámbitos del dibujo y el diseño. Su razón de ser es revalorizar el arte" \
" hecho por y para las personas." , text_alignment="center")
st.text("  El sistema y su base de datos" \
" se han inspirado en la colección online" \
" del Museo Nacional del Prado, que puede visitarse completa y tiene todas sus" \
" piezas disponibles para descarga en la mejor calidad. ", text_alignment="center")


# Inicializamos elementos
db_cursor = Utils.get_db_connection()
db_cursor_raw =Utils.get_db_connection_raw()
resnet50_model = Utils.initialize_model()
resnet50_transform = Utils.create_pipeline(resnet50_model)
Image.MAX_IMAGE_PIXELS = None # Cambiamos configuración por defecto de librería Image para que no de errores de resolución
museums_coordinates = {
    "Museo del Prado": {
        "lat": 40.4138,
        "lon": -3.6921
    },
    "Museo Nacional Thyssen-Bornemisza": {
        "lat": 40.4160,
        "lon": -3.6946
    },
    "Museo Nacional Centro de Arte Reina Sofía": {
        "lat": 40.4087,
        "lon": -3.6942
    },
    "Museo Picasso Barcelona": {
        "lat": 41.3853,
        "lon": 2.1800
    },
    "Museo de Montserrat": {
        "lat": 41.5933,
        "lon": 1.8370
    },
      "Museo Sorolla": {
        "lat": 40.4345,
        "lon": -3.6923
    },
    "Centro Pompidou": {
        "lat": 48.8606,
        "lon": 2.3522
    }
}

# Muestra una imagen con marco
def show_Image(blob):
    image = Image.open(io.BytesIO(blob))
    bordered_image = ImageOps.expand(image, border=20, fill='#2E2B2C')
    st.image(bordered_image)

# Obtención de la imagen del usuario
new_image = st.file_uploader(
    label="Subir una imagen para explorar cuadros similares:",
    type=["png", "jpeg", "jpg"])

# Si la imagen nueva es válida
if new_image is not None:
    new_blob = new_image.read() # devuelve la imagen en bytes (blob)
    show_Image(new_blob)
    # Vectorizar la imagen del usuario
    new_embedding = Utils.get_embedding(new_blob,resnet50_model,resnet50_transform)

    # Contenedores para separar claramente los resultados con un borde
    container1 = st.container(border=True)  
    container2 = st.container(border=True)
    container3 = st.container(border=True)
    containers = [container1, container2, container3]

    # Sistema de recomendación con KDTree
    tree, painting_names = Utils.build_kdtree(db_cursor)
    closest_paintings= Utils.get_closest_paintings_tree(new_embedding, tree, painting_names, k=3)
    # Recorremos los resultados y mostramos cada cuadro y su información en un contenedor separado
    for i, (painting, container) in enumerate(zip(closest_paintings, containers), 1):
        with container:
            creator, museum, movement = Utils.get_painting_creator_museum_movement(painting, db_cursor)
            if i == 1:
                col1, col2 = st.columns([0.77, 0.23])
                with col2:
                    st.badge(label="Mejor resultado", color="green")
                with col1:
                    st.markdown(f'''
                        #### {i}. {painting}
                    de {creator}, pertenece al {movement}
                    ''')
            else :
                st.markdown(f'''
                            #### {i}. {painting}
                        de {creator}, pertenece al {movement}
                        ''')
            painting_url = Utils.get_wikidata_ref(painting,db_cursor,db_cursor_raw)[0]
            # Mostrar la imagen
            painting_blob = Utils.get_blob(painting,db_cursor)
            show_Image(painting_blob)
            #
            st.write(f"[Ficha en Wikidata de \"{painting}\"]({painting_url})")
            museum_lat = museums_coordinates[museum]["lat"]
            museum_lon = museums_coordinates[museum]["lon"]
            museum_coord = pd.DataFrame({
            'lat': [museum_lat],
            'lon': [museum_lon]
        })
            st.write(f"Esta obra está localizada actualmente en el {museum}, mostrado en el mapa:")
            st.map(museum_coord)


st.text("Todos los cuadros de la base de datos de este sistema están incluidos" \
" en una de estas colecciones. Haga click en las flechas para ver todos los museos:", text_alignment="center")
from streamlit_carousel import carousel
carousel_items = [
    dict(
        title="Museo Nacional del Prado",
        text="Haga click aquí para ver la página oficial del museo",
        img="C:/Users/evrio/Desktop/TFG/App/Images/Museo_del_Prado_(Noordelijke_ingang)_2.jpg",
        link="https://www.museodelprado.es/",
    ),
    dict(
        title="Museo Nacional Thyssen-Bornemisza",
        text="Haga click aquí para ver la página oficial del museo",
        img="C:/Users/evrio/Desktop/TFG/App/Images/Museo_Thyssen-Bornemisza.jpg",
        link="https://www.museothyssen.org/",
    ),
    dict(
        title="Museo Nacional Centro de Arte Reina Sofía",
        text="Haga click aquí para ver la página oficial del museo",
        img="C:/Users/evrio/Desktop/TFG/App/Images/Madrid_-_Extensión_del_Museo_Nacional_Centro_de_Arte_Reina_Sofía_(MNCARS)_01.jpg",
        link="https://www.museoreinasofia.es/",
    ),
    dict(
        title="Museo Sorolla",
        text="Haga click aquí para ver la página oficial del museo",
        img="C:/Users/evrio/Desktop/TFG/App/Images/Mujeres_en_la_playa,_Joaquín_Sorolla.jpg",
        link="https://www.cultura.gob.es/msorolla/inicio.html",
    ),
    dict(
        title="Centre Pompidou",
        text="Haga click aquí para ver la página oficial del museo",
        img="C:/Users/evrio/Desktop/TFG/App/Images/Centre_Pompidou,_Paris._-_Flickr_-_Lejeune_Grégory.jpg",
        link="https://www.centrepompidou.fr/es/pompidou",
    ),
    dict(
        title="Museo Picasso de Barcelona",
        text="Haga click aquí para ver la página oficial del museo",
        img="C:/Users/evrio/Desktop/TFG/App/Images/Exposició_Picasso_i_Raventós.jpg",
        link="https://museupicassobcn.cat/es/",
    ),
    dict(
        title="Museo de Montserrat",
        text="Haga click aquí para ver la página oficial del museo",
        img="C:/Users/evrio/Desktop/TFG/App/Images/Monistrol_de_Montserrat_-_Monasterio_de_Santa_María_de_Montserrat_73.jpg",
        link="https://www.museudemontserrat.com/ca/index.html",
    ),
]
carousel(items=carousel_items)

credits = st.container(border=True)
with credits:
    st.text("Todas las imágenes del carrusel de museos tienen licencias sin restricciones de uso y han sido obtenidas de wikimedia.",text_alignment="center")