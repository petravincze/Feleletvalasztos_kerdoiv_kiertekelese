# USAGE
# Feleletvalasztos kerdoiv kiertekelese

# szukseges python csomagok importalasa
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#Elso resz - a kep elokeszitese

# kiereteklendo kep megadasa a program futtatasakor
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

#valtozok megadasa
#bool circle = false #az alakzat kor-e
#bool square = false #az alakzat negyzet-e
var answers = 0 #jelolt valaszok darabszama
var question = 1

#kep betoltese
#szurkearnyalatos konvertalas
#magas frekvenciaju zajok csokkentese
#elek megtalalasa
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# konturok megkeresese az elterkepen
#a dokumnetum konturjainak inicializalasa
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

#ellenorzes, hogy min egy kontur van
if len(cnts) > 0:
    #konturok csokkeno sorrendbe rendezese
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # vegigmegyunk a rendezett konturokon
    for c in cnts:
        #konturok kozelitese
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        #ha 4 pontot talalunk, akkor beazonositottuk a papirt
        if len(approx) == 4:
            docCnt = approx
            break

#alkalmazzunk four point perspective transzformaciot a felulnezeti kepert
#az erdetei kepen
paper = four_point_transform(image, docCnt.reshape(4, 2))
#a szurkearnyalatos kepen
warped = four_point_transform(gray, docCnt.reshape(4, 2))

#Masodik resz - kerdesek es valaszok azonositasa
#egyenlore csak kort kepes felismerni, a negyzetek felismerese hianyzik

#Otsu metodussal alakitsuk at a kepet binaris keppe
#elkulonul a hatter es a kerdesek, valaszok
thresh = cv2.threshold(warped, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#konturok megkeresese a binaris kepen
#valtozo inicializalasa a kerdesek konturjanak
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

#vegigmegyunk a konturokon
for c in cnts:
    #konturok hatarertekinek megadasa
    #lehetove teszi a keparany kiszamitasat
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    #annak megallapitasa, hogy a kontur kerdest jelol-e
    #megfelelol szelessegre es magassagra van szukseg
    #keparanynak 1 korulinek kell lennie
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# fentrol lefele rendezes
questionCnts = contours.sort_contours(questionCnts,
    method="top-to-bottom")[0]

#valaszlehetosegek szamanak inicializalasa
#possibleAnswers = 4

# minden kerdesnel 4 valaszlehetoseg van megadva
#ezert negyesevel megyunk vegig a tombon
for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
    #kerdesek rendezese balrol jobbra
    cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
    #a jelolt valasz inicializalasahoz valtozo
    bubbled = None

    #vegigmegyunk a konturokon
    for (j, c) in enumerate(cnts):
        #keszitsunk maszkot, ami az eppen aktualis valaszlehetoseget mutatja
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        #alkalmazzuk a maszkot a binaris kepen
        #szamoljuk ossze a nem-nulla pixeleket a teruleten
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        #ha az aktualis ertek nagyobb, mint az eddig meghatarozott legnagyobb
        #akkor legyen ez az uj maximalis ertek
        #a jelolt valaszok szamat noveljuk eggyel
        
        #esetleges modositas: ha egy bizonyos ertekkel kisebb vagy nagyobb, mint a maximalis,
        #akkor is noveljuk a jelolt valaszok szamat
        
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
            answers += 1
    #kor eseteben a valaszlehetosegek szama nem lehet egynel tobb
    #irjuk ki, hogy ervenytelen valaszadas
    if answers > 1:
        printf ('A %d. kerdesre hibas valasz erkezett' %question)
        answers = 0
    
    #a kerdes szamat noveljuk 1-gyel
    question += 1

cv2.waitKey(0)
