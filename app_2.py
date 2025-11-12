from fastapi import FastAPI, file, uploadedfile, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import select
from io import StringIO
import pandas as pd
from joblib import load
import pytz
from datetime import datetime
import os




SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:jbjpqMhkwrDmteyDakQvmbXZwlIPygvq@maglev.proxy.rlwy.net:57475/railway"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
metadata = MetaData()


items = Table("items", metadata, autoload_with=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Item(BaseModel):
    id: int
    name: str
    description: str = None

    class Config:
        from_attributes = True

class ItemCreate(BaseModel):
    name: str
    description: str = None

class ItemUpdate(ItemCreate):
    pass


app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/items/{item_id}", response_model=Item)
def read_item(item_id: int):
    with SessionLocal() as session:
        query = select(items).where(items.c.id == item_id)
        db_item = session.execute(query).fetchone()
        if db_item is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
        return db_item._mapping

@app.post("/items/", response_model=Item)
def create_item(item: ItemCreate):
    with SessionLocal() as session:
        new_item = items.insert().values(name=item.name, description=item.description)
        result = session.execute(new_item)
        session.commit()
        created_item_id = result.inserted_primary_key[0]
        created_item = session.execute(select(items).where(items.c.id == created_item_id)).fetchone()
        return created_item._mapping

@app.put("/items/{item_id}", response_model=Item)
def update_item(item_id: int, item: ItemUpdate):
    with SessionLocal() as session:
        update_values = {k: v for k, v in item.dict().items() if v is not None}
        updated_item = items.update().where(items.c.id == item_id).values(**update_values)
        session.execute(updated_item)
        session.commit()
        updated_item = session.execute(select(items).where(items.c.id == item_id)).fetchone()
        return updated_item._mapping

@app.delete("/items/{item_id}", response_model=Item)
def delete_item(item_id: int):
    with SessionLocal() as session:
        query = select(items).where(items.c.id == item_id)
        deleted_item = session.execute(query).fetchone()
        if deleted_item is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
        session.execute(items.delete().where(items.c.id == item_id))
        session.commit()
        return deleted_item._mapping

@app.get("/items/search/{query}", response_model=list[Item])
def search_items(query: str):
    with SessionLocal() as session:
        query = select(items).where(items.c.name.ilike(f"%{query}%"))
        results = session.execute(query).fetchall()
        return [result._mapping for result in results]

@app.get("/items/", response_model=list[Item])
def get_items(skip: int = 0, limit: int = 10):
    with SessionLocal() as session:
        query = select(items).offset(skip).limit(limit)
        results = session.execute(query).fetchall()
        return [result._mapping for result in results]


app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health",status_code=200,include_in_schema=False)
def health_check(db= Depends(get_db)):
    return {"status": "ok"}



@app.post("/predict/")
def predict_house(file: uploadedfile.UploadedFile = file(...),db: Session = Depends(get_db)):
    
    # Load the pre-trained model
    classifier = load("linear_regression.joblib")

    
    features_df = pd.read_csv("selected_features.csv")
    features = features_df['0'].to_list()

    content = await file.read()

    df = pd.read_csv(StringIO(content.decode("utf-8")))
    data = df[features]

    predictions =classifier.predict(df)

    colombia_tz = pytz.timezone("America/Bogota")
    now = datetime.now(colombia_tz)
    
    predictions_df = pd.DataFrame(
        {   
            "file_name": file.filename,
            "prediction": predictions,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
        }
    )

    predictions_df.to_sql("predictions", con=engine, if_exists="append", index=False)



    return {"predictions": predictions.tolist()
    }