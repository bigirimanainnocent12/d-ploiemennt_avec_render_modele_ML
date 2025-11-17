from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from enum import Enum
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="API de déploiement du modèle de machine learning",
    description="Cette API permet d’estimer les charges d’assurance santé à partir de données personnelles et comportementales du client. Elle prend en entrée des variables telles que l’âge, le sexe, l’indice de masse corporelle (IMC), le nombre d’enfants à charge, le statut de fumeur et la région de résidence. En sortie, elle retourne une estimation des frais médicaux ou des primes à payer, calculée par un modèle de machine learning entraîné sur des données d’assurance.",
    version="1.0",
    contact={
        "name": "Innocent BIGIRIMANA",
        "email": "gatoziinnocent@gmail.com",
        "url": "https://bigirimanainnocent12.github.io/PORTFOLIO/"
    }

)


# Monter le dossier statique
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route pour servir le favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

class Sexe(str, Enum):
    Homme = "Homme"
    Femme = "Femme"

class Smoker(str, Enum):
    Yes = "Yes"
    Non = "Non"

class Region(str, Enum):
    northeast = "Nord"
    northwest = "Sud"
    southeast = "Est"
    southwest = "Ouest"


class Caracteristique(BaseModel):
    age: int
    sexe: Sexe
    bmi: float
    children: int
    smoker: Smoker
    region: Region




def lire_model(modele):
    loaded_model = joblib.load(modele)
    if loaded_model is None:
        return "modele non load"
    else:
        return loaded_model

   


@app.get("/",tags=["info"])

def info():
    return {

   "message":"API PROJET COMPLET"
    }


@app.get("/deploiement/", tags=["info"])
def deploiement(age: int, sexe: Sexe, bmi: float, children: int, smoker: Smoker, region: Region):
    
    dicte = {
        "age": [age],
        "sex": [sexe.value],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker.value],
        "region": [region.value]
    }

    

    campagne = pd.DataFrame(dicte)
    campagne['sex'] = campagne['sex'].map({'Homme': True, 'Femme': False})
    campagne['smoker'] = campagne['smoker'].map({'Yes': True, 'Non': False})

    modele = lire_model("modele.pkl")
    prediction = modele.predict(campagne)[0]

    return {"Sa charge est de ": prediction}


    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application:app", host="localhost", port=8000)
