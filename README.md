# Duckie Town (Deep Learning homework)

"Deep Learning a gyakorlatban Python és LUA alapon" (VITMAV45)

## Adatok

Csapatnév: AgentP

Csapattagok: 
- Fazekas Lajos
- Kozák Aron
- Szász Zsolt

## Tartalom

**1.** Egy kézzel irányított agent adatainak kinyerését megvelósító program: ```imitation_learning.py``` 

**2.** Egy az adatok előkészítését végző program: ```prep.py```

**3.** Három mappa található a repoban (_conv,lstm,q_) ezek mindegyike tartalmazza a következőket:

- a model-t definiáló program:   ```MODELNEV_model.py```
- a model-t trainelni képes program:   ```MODELNEV_train_TRAINTIPUS.py```
- a model általi futtatást megvalósító program:   ```MODELNEV_play.py```

(a MODELNEV helyére a _conv,lstm,q_ hármasból lehet választani)

## Használat

A "MODELNEV" helyére a _conv,lstm,q_ hármasból lehet választani mindenhol.

A környezet felállítása:
```
git clone https://github.com/duckietown/gym-duckietown.git
cd gym-duckietown
pip3 install -e .
git clone https://github.com/zsoca000/Duckie-Town-DL-Homework
cd Duckie-Town-DL-Homework
pip3 install -e .
mv conv/* .
mv lstm/* .
mv q/* .
rm -rf conv
rm -rf lstm
rm -rf q
```

A meglévő háló által irányított agent indítása:
```
python3 MODELNEV_play.py
```


A meglévő hálók továbbtanítása:
(itt nem ajánlott a fit-et használni mert nagy adathalmazra nem működik)
```
python3 MODELNEV_train_on_batch.py
```



Új háló tanítása:
```
python3 MODELNEV_train_on_batch.py --load False
```

Új adatok gyártása az imitation learning alapú modelleknek (_lstm,conv_).
VIGYÁZAT ez törli az eddigieket. Kell játszani vele pár percig:
```
rm datas/x/*
rm datas/y/*
rm names.npy
python3 imitaton_learning.py
```


## Korrigálás:

Egy dolgot nem tudtunk megoldani. A pretrained modelleket és az általunk gyártott tanító adatokat méretük miatt nem tudtuk feltölteni a repoba ezekhez itt egy link:

https://bmeedu-my.sharepoint.com/:f:/g/personal/fazekas_lajos_edu_bme_hu/EmZwGOGwwA5LlgESh5elLHoBuBGWBf1BCBuZZuZMQ45wVA?e=RuyWsP. 

Az a fontos, hogy a Duckie-Town-DL-Homework mappában kell legyen egy datas mappa és a pretrained modell mappája illetve a tanító adatok mappái (x,y), bármilyen őket felhasználó program esetén kell létezzenek.

