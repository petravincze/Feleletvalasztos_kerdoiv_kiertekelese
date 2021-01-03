# USAGE
# Feleletvalasztos kerdoiv kiertekelese

# szukseges python csomagok importalasa
from imutils.perspective import four_point_transform
from imutils import contours
from alakzat import Shape
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

#Valtozok inicializalasa
#kerdesek szama
question = 1
#jelolt valaszok darabszama
answers = 0
#a kerdoiv helyesen van-e kitoltve
helyes = True

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

#szukseg van-e four point perspective transzformaciora
kepe = input("Ha a kerdoiv valamilyen eszkozzel lett lefotozva, nyomja meg az 1-es billentyut, ha szkennelve van, nyomja meg a 0-as billentyut: ")

#ha lefotozott papir a kep
if kepe == '1':
    #alkalmazzunk four point perspective transzformaciot a felulnezeti kepert
    #az erdetei kepen
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    #a szurkearnyalatos kepen
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    
    #Otsu metodussal alakitsuk at a kepet binaris keppe
    #elkulonul a hatter es a kerdesek, valaszok
    thresh = cv2.threshold(warped, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
elif kepe== '0':
    #Otsu metodussal alakitsuk at a kepet binaris keppe
    #elkulonul a hatter es a kerdesek, valaszok
    thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
else:
    print ('Hibas gomb')

#Masodik resz - kerdesek es valaszok azonositasa

#konturok megkeresese a binaris kepen
#valtozo inicializalasa a kerdesek konturjanak
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []
q = 0

#valtozo inicializalasa az alakzat azonositasahoz
s = Shape()

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
        q += 1

# fentrol lefele rendezes
questionCnts = contours.sort_contours(questionCnts,
    method="top-to-bottom")[0]


#valaszlehetosegek szamanak inicializalasa
#possibleAnswers = input('Egy kerdeshez tartozo valaszok szama: ')
#answersCount = int(possibleAnswers)

#ha nem ugyanannyi valasz tartozik egy kerdeshez
answersCount = [] 
#kerdesek szama
print ('%d alakzatot talalt' %q)
n = int(input("Hany kerdes van a kerdoivben : ")) 
#valaszok szamanak eltarolasa
for i in range(0, n): 
    a = int(input('Adja meg az %d. kerdeshez tartozo valaszok szamat! ' %question)) 
  
    #ertek hozzadasa a listahoz
    answersCount.append(a)
    
    question += 1

n = 0
question = 1

#vegigmegyunk a tombon, akkora leptekkel amekkorat a felhasznalo megadott
for (q, i) in enumerate(np.arange(0, len(questionCnts), answersCount[n])):
    #kerdesek rendezese balrol jobbra
    cnts = contours.sort_contours(questionCnts[i:i + answersCount[n]])[0]
    
    #logikai valtozo inicializalasa a valaszokhoz tartozo alakzat megallapitasahoz
    negyzet = False
    #valtozo inicializalasa ahhoz, hogy tudjuk, a kerdeshez tartozo elso alakzatot vizsgaljuk-e
    hanyadik_valasz = 1

    #vegigmegyunk a konturokon
    for (j, c) in enumerate(cnts):
        
        #megnezzuk, hogy az alakzat negyzet-e vagy sem
        #eleg csak minden kerdesnel az elso alakzatot megvizsgalni
        while hanyadik_valasz == 1:
            
            alakzat = s.detect(c)
            #print (alakzat)
            if alakzat == "negyzet":
                negyzet = True
            
            hanyadik_valasz += 1
            
        #keszitsunk maszkot, ami az eppen aktualis valaszlehetoseget mutatja
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        #alkalmazzuk a maszkot a binaris kepen
        #szamoljuk ossze a nem-nulla pixeleket a teruleten
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        
        #adjunk meg egy erteket ami felett valoszinuleg jelolve van a valaszlehetoseg
        print (total)
        if total > 250:
            answers += 1
            #print('valasz')

    #kor eseteben a valaszlehetosegek szama nem lehet egynel tobb
    #negyzet eseteben tobb valasz is jelolheto
    if negyzet == False:
        if answers > 1 or answers == 0:
            print ('A %d. kerdes hibasan lett kitoltve' %question)
            helyes = False
    else:
        if answers == 0:
            print ('A %d. kerdes hibasan lett kitoltve' %question)
            helyes = False
    
    #a kerdes szamat noveljuk 1-gyel
    print(question)
    question += 1
    #valaszok szama 0 legyen
    answers = 0
    #n erteket noveljuk
    n += 1

#ha a valtozo erteke igaz, akkor a kitoltesben nem volt hiba
if helyes == True:
    print ('A kerdoiv helyesen lett kitoltve!')

cv2.waitKey(0)