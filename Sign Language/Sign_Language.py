# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 01:07:50 2024

@author: sarib
"""

import cv2 # OpenCV kütüphanesi, görüntü işleme ve video yakalama işlemleri için kullanılır
import numpy as np
import os # İşletim sistemi ile etkileşimde bulunmak için kullanılır 
from matplotlib  import pyplot as plt 
import time #  Zamanla ilgili işlemler için kullanılır
import mediapipe as mp #  Gelişmiş görüntü işleme ve makine öğrenimi modelleri için bir framework.


#%% #hücrelere bölmek için bu işareti kullan ve seçili hücreyi çalıştırmak için ctr+enter a bas

mp_holistic = mp.solutions.holistic # modülü, MediaPipe Holistic modelini kullanarak yüz, el ve vücut duruşunu (pose) tespit edebilen bir çözümdür. Bu modül, insanların kompleks hareketlerini, vücut pozisyonlarını ve hareketlerini tek bir kolay kullanımlı çözümle algılar ve analiz eder.
#Holistic model, yüz landmark'larını, el landmark'larını ve pose landmark'larını (vücut duruşunu belirleyen noktalar) aynı anda algılayarak, daha bütünleşik ve ayrıntılı bir insan hareketi analizi sunar.
mp_drawing = mp.solutions.drawing_utils # modülü, algılanan landmark'lar ve diğer algılama sonuçlarını görselleştirmek için kullanılır. Bu modül, belirlenen landmark noktalarını ve bağlantıları çizme, böylece elde edilen algılama sonuçlarını anlaşılır bir şekilde görsel olarak sunma işlevine sahiptir.

def mediapipe_detection(image,model): #image: İşlem yapılacak olan görüntü (genellikle bir video karesi) , model: MediaPipe tarafından sağlanan bir görüntü işleme modeli (örneğin, yüz, el tanıma veya poz tahmini modeli).
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)#cv2.cvtColor: OpenCV'nin renk dönüşüm fonksiyonudur. Burada image isimli görüntü, BGR (Blue, Green, Red) renk formatından RGB (Red, Green, Blue) renk formatına dönüştürülür. MediaPipe modelleri genellikle RGB formatında görüntü işler.
    image.flags.writeable= False # görüntünün "writeable" (yazılabilir) bayrağını False olarak ayarlar. Bu, modelin görüntü üzerinde yerinde değişiklik yapmasını engeller ve bazı durumlarda performansı artırabilir.
    results = model.process(image) # Belirtilen model kullanılarak image üzerinde işlem yapılır. Bu işlem, modelin görüntüden çeşitli özellikler çıkarmasını sağlar (örneğin, yüz landmark'ları, el pozisyonları vb.).
    image.flags.writeable = True # Görüntünün "writeable" (yazılabilir) bayrağı tekrar True olarak ayarlanır. Bu, işlemden sonra görüntü üzerinde değişiklik yapılabilmesine izin verir.
    image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR) # İşlemler tamamlandıktan sonra, görüntüyü tekrar OpenCV'nin varsayılan renk formatı olan BGR'ye dönüştürür. Bu, sonraki OpenCV işlemleri için uygun olacaktır.
    return image , results  # işlenmiş görüntüyü (image) ve model tarafından elde edilen sonuçları (results) döndürür. Bu sonuçlar genellikle görüntüde tespit edilen özelliklerin veya nesnelerin koordinatlarını içerir.
'''
Bu fonksiyon, MediaPipe ile görüntü işleme iş akışını standartlaştırır ve görüntü işleme sürecini daha yönetilebilir 
ve tekrar kullanılabilir hale getirir. Fonksiyon, herhangi bir MediaPipe modeli ile kullanılabilir ve çeşitli uygulamalarda 
kolaylıkla entegre edilebilir.
'''
'''aşşağıda color kısmındaki sayılar BGR(RGB nin tersi) e göre ayarlanıyor ''' 
def draw_landmarks(image, results): # Bu fonksiyon, işlenmiş görüntüyü (image) ve MediaPipe'dan alınan sonuçları (results) parametre olarak alır. tespit edilen vücut, yüz ve el landmark'larını (anahtar noktalarını) belirli bağlantılarla çizmek için kullanılır.
    if results.face_landmarks:#  Yüz landmark'larının (face_landmarks) var olup olmadığını kontrol eder.
        mp_drawing.draw_landmarks( #  Tespit edilen landmark'ları çizer. 
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,# results.face_landmarks: Çizilecek yüz landmark'ları. , mp_holistic.FACEMESH_TESSELATION: Yüz landmark'larını birbirine bağlayan çizgi bağlantılarını tanımlar.
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1), # landmark_drawing_spec: Tek tek landmark noktalarının nasıl çizileceğini tanımlar (renk, kalınlık, çember boyutu).
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,127), thickness=1, circle_radius=1) # connection_drawing_spec: Landmark'lar arasındaki bağlantı çizgilerinin nasıl çizileceğini tanımlar.
        )

    if results.pose_landmarks: #  Vücut duruşu landmark'larının (pose_landmarks) var olup olmadığını kontrol eder.
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, # mp_holistic.POSE_CONNECTIONS: Vücut duruşu landmark'larını birbirine bağlayan çizgi bağlantılarını tanımlar
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,# mp_holistic.HAND_CONNECTIONS: El landmark'larını birbirine bağlayan çizgi bağlantılarını tanımlar.
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,# mp_holistic.HAND_CONNECTIONS: El landmark'larını birbirine bağlayan çizgi bağlantılarını tanımlar.
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

'''
min_detection_confidence:
Bir nesnenin ilk kez algılandığında geçerli sayılması için gereken minimum güven skoru.
Algılama (detection) aşamasında kullanılır.
İlk tespitin doğruluğunu belirler.

min_tracking_confidence:
Algılanmış bir nesnenin izlenmesi sırasında geçerli sayılması için gereken minimum güven skoru.
İzleme (tracking) aşamasında kullanılır.
İzleme işleminin doğruluğunu belirler.

'''

#with anahtar kelimesi, bir context manager ile kullanıldığında, bir kaynağın yönetimini otomatize etmek ve temizleme kodunu daha güvenilir ve okunabilir hale getirmek için kullanılır. Context managerlar, Python'daki with ifadesiyle birlikte çalışacak şekilde tasarlanmış özel nesnelerdir. Bu yapı, çeşitli kaynakların (dosyalar, ağ bağlantıları, veritabanı bağlantıları vb.) açılıp kapatılmasını kolaylaştırır.
cap = cv2.VideoCapture(0) # Bilgisayarınızın varsayılan kamerasını aktif hale getirir (0 genellikle varsayılan kamerayı işaret eder).
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
   while cap.isOpened(): # Kamera açık olduğu sürece döngü devam eder.
        ret , frame = cap.read() # Kameradan bir kare (frame) yakalar. ret, kare yakalamanın başarılı olup olmadığını (True/False) belirten bir boolean değerdir.
        
        image , results1 = mediapipe_detection(frame,holistic) # Yani, image değişkeni işlenmiş frame değerini alır ve results değişkeni holistic modelinin işlem sonucunu alır
        
        draw_landmarks(image, results1) # işlenmiş frame i ve sonuçları draw_landmarks foksiyonuna göndeririz
        
        cv2.imshow('OpenCV Feed', image) #Yakalanan kareyi bir pencerede gösterir. Pencere başlığı 'OpenCV Feed' olarak ayarlandı.
        #cv2.imshow fonksiyonu, OpenCV kütüphanesinin bir parçasıdır ve bu fonksiyon, bir görüntüyü belirli bir pencere üzerinde göstermek için kullanılır.
        if cv2.waitKey(10) & 0xFF == ord('q'): # kamera q ya basınca kapanır
            #Her 10 milisaniyede bir, klavyeden bir tuşa basılıp basılmadığını kontrol eder. Eğer 'q' tuşuna basılırsa, döngüyü kırarak görüntü yakalama işlemini sonlandırır.
            '''
            & İşareti:
            & işareti, bit düzeyinde bir AND işlemidir. Yani, iki sayının bitleri karşılaştırılır ve her iki sayının ilgili bitleri de 1 ise, sonuçtaki bit 1 olur; aksi takdirde 0 olur. Bu işlem, belirli bit maskelerini uygulamak için kullanılır.
            0xFF:
            0xFF hexadecimal sistemde 255 demektir ve binary olarak 11111111 olarak temsil edilir. Bu, 8 bitlik bir maske anlamına gelir.
            cv2.waitKey(10) fonksiyonu, belirtilen süre (milisaniyeler cinsinden) boyunca bir klavye tuşuna basılıp basılmadığını kontrol eder ve basılan tuşun ASCII değerini bir tam sayı olarak döndürür.
            Fakat cv2.waitKey 32 bitlik bir değer döndürür ve genellikle bu değerin son 8 biti tuş bilgisini içerir. & 0xFF işlemi, döndürülen değerin sadece son 8 bitini (yani gerçek tuş değerini) elde etmek için kullanılır.
            İfade Anlamı:
            cv2.waitKey(10) & 0xFF, cv2.waitKey tarafından döndürülen değerin son 8 bitini maskeleyerek, gerçek tuş değerini çıkarır.
            ord('q') ise, 'q' karakterinin ASCII değerini verir. ASCII tablosunda 'q' harfinin değeri 113'tür.
            Dolayısıyla, if cv2.waitKey(10) & 0xFF == ord('q') ifadesi, eğer kullanıcı son 10 milisaniye içinde 'q' tuşuna basmışsa True değerini döndürür ve bu koşul sağlandığında döngü kırılır (program kamera görüntüsünü göstermeyi durdurur).
            '''
            break
   cap.release() # cv2.VideoCapture tarafından başlatılan video yakalama cihazını (örneğin, bir kamera) serbest bırakır. Bu fonksiyon çağrıldığında, OpenCV video yakalama cihazı ile olan bağlantıyı keser. Böylece, kamera gibi donanım kaynakları başka uygulamalar tarafından kullanılabilir hale gelir.
   cv2.destroyAllWindows()# OpenCV ile açılan tüm pencereyi kapatır. Eğer programınız bir veya birden fazla pencere oluşturduysa (örneğin, cv2.imshow() fonksiyonu ile görüntü gösterimi yapılıyorsa), bu fonksiyon tüm bu pencereleri kapatır ve kaynakları temizler.
   

#%%

def extract_keypoint(results):
    pose = np.array([[res.x,res.y,res.z , res.visibility] for res in results1.pose_landmarks.landmark]).flatten() if results1.pose_landmarks else np.zeros(33*4) #Bu ifade, results1.pose_landmarks mevcutsa (vücut pozisyonu landmarkları tespit edilmişse), her bir landmark için x, y, z koordinatlarını ve visibility (görünürlük) değerini içeren bir numpy dizisi oluşturur , mevcut değilse diziye 0 atar
    print(pose)
    
    print(len(pose)) # pose dizisinin boyutu , bu ifade ile yukarıdaki kısımdaki np zeros kısmının sayısal degerinin kaç olacağını görebiliriz
    print(pose.shape) # pose dizisinin kaç satır ve sutundan oluştuğu

    face = np.array([[res.x,res.y,res.z] for res in results1.face_landmarks.landmark]).flatten() if results1.face_landmarks else np.zeros(468*3) # Bu ifade, eğer left_hand_landmarks mevcutsa yüz landmarkları için x, y, z değerlerinden oluşan bir numpy dizisi oluşturur ve flatten() ile tek boyuta indirger.
    print(face) # Eğer yüz landmarkları mevcut değilse, 468*3 boyutunda bir sıfır dizisi oluşturulur (yüz modelinde 468 nokta, her biri için 3 değer).

    lh = np.array([[res.x,res.y,res.z] for res in results1.left_hand_landmarks.landmark]).flatten() if results1.left_hand_landmarks else np.zeros(21*3) # Eğer sol el landmarkları mevcutsa, bu landmarklar için x, y, z değerlerini içeren bir numpy dizisi oluşturur ve flatten() ile tek boyuta indirger
    print(lh)

    rh = np.array([[res.x,res.y,res.z] for res in results1.right_hand_landmarks.landmark]).flatten() if results1.right_hand_landmarks else np.zeros(21*3) # Eğer sağ el landmarkları mevcutsa, bu landmarklar için x, y, z değerlerini içeren bir numpy dizisi oluşturur ve flatten() ile tek boyuta indirger.
    print(rh)
    return np.concatenate([pose,face,lh,rh]) # Tüm bu dizi parçalarını (pose, face, lh, rh) tek bir numpy dizisinde birleştirir.

print(extract_keypoint(results1).shape)

result_test = extract_keypoint(results1)
np.save('0',result_test)
np.load('0.npy')
#%%

DATA_PATH = os.path.join("MY_DATA") #  Bu fonksiyon, verilen yolları birleştirir ve işletim sistemine uygun bir dosya yolu oluşturur. Bu sayede, farklı işletim sistemlerinde dosya yollarıyla çalışırken uyumluluk sorunlarını önlemiş olursunuz.
actions = np.array(['hello','iloveyou','thanks'])
no_sequences = 10 
sequence_length = 10

for action in actions:
    for sequence in range(no_sequences):
        try:# Try-Except Bloğu: Kodun bu kısmı, hata yönetimi için kullanılır. Klasör oluşturma işlemi sırasında herhangi bir hata meydana gelirse, except bloğu çalışır ve bu hata yoksayılır (yani herhangi bir işlem yapılmaz).
            os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))#os.makedirs fonksiyonu, verilen yolda klasör(ler) oluşturur. Eğer klasör zaten varsa ve exist_ok=False (varsayılan) ise bir hata fırlatır. Bu fonksiyon, iç içe klasörler oluşturabilir.
        except:# os.path.join(DATA_PATH, action, str(sequence)) ifadesi,DATA_PATH dizini altında, her bir action için, sequence numarasını içeren klasörlerin yollarını birleştirir. str(sequence) ile sequence numarası stringe çevrilir çünkü dosya yolu bir string olmalıdır.
            pass


#%%

#with anahtar kelimesi, bir context manager ile kullanıldığında, bir kaynağın yönetimini otomatize etmek ve temizleme kodunu daha güvenilir ve okunabilir hale getirmek için kullanılır. Context managerlar, Python'daki with ifadesiyle birlikte çalışacak şekilde tasarlanmış özel nesnelerdir. Bu yapı, çeşitli kaynakların (dosyalar, ağ bağlantıları, veritabanı bağlantıları vb.) açılıp kapatılmasını kolaylaştırır.
cap = cv2.VideoCapture(0) # Bilgisayarınızın varsayılan kamerasını aktif hale getirir (0 genellikle varsayılan kamerayı işaret eder).
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
   
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                
                    ret , frame = cap.read() # Kameradan bir kare (frame) yakalar. ret, kare yakalamanın başarılı olup olmadığını (True/False) belirten bir boolean değerdir.
                    
                    image , results1 = mediapipe_detection(frame,holistic) # Yani, image değişkeni işlenmiş frame değerini alır ve results değişkeni holistic modelinin işlem sonucunu alır
                    
                    draw_landmarks(image, results1) # işlenmiş frame i ve sonuçları draw_landmarks foksiyonuna göndeririz
                    
                    if frame_num == 0:
                        cv2.putText(image,'STARTING COLLECTİON',(120,200),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                        cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,sequence),(15,12),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                        cv2.waitKey(1000)
                    else:
                        cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,sequence),(15,12),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                    
                    keypoints = extract_keypoint(results1)
                    npy_path = os.path.join(DATA_PATH,action,str(sequence),str(frame_num))
                    np.save(npy_path,keypoints)
                    
                    cv2.imshow('OpenCV Feed', image) #Yakalanan kareyi bir pencerede gösterir. Pencere başlığı 'OpenCV Feed' olarak ayarlandı.
                    #cv2.imshow fonksiyonu, OpenCV kütüphanesinin bir parçasıdır ve bu fonksiyon, bir görüntüyü belirli bir pencere üzerinde göstermek için kullanılır.
                    if cv2.waitKey(10) & 0xFF == ord('q'): # kamera q ya basınca kapanır
                                #Her 10 milisaniyede bir, klavyeden bir tuşa basılıp basılmadığını kontrol eder. Eğer 'q' tuşuna basılırsa, döngüyü kırarak görüntü yakalama işlemini sonlandırır.
                          break
    cap.release() # cv2.VideoCapture tarafından başlatılan video yakalama cihazını (örneğin, bir kamera) serbest bırakır. Bu fonksiyon çağrıldığında, OpenCV video yakalama cihazı ile olan bağlantıyı keser. Böylece, kamera gibi donanım kaynakları başka uygulamalar tarafından kullanılabilir hale gelir.
    cv2.destroyAllWindows()# OpenCV ile açılan tüm pencereyi kapatır. Eğer programınız bir veya birden fazla pencere oluşturduysa (örneğin, cv2.imshow() fonksiyonu ile görüntü gösterimi yapılıyorsa), bu fonksiyon tüm bu pencereleri kapatır ve kaynakları temizler.
   

cap.release() 
cv2.destroyAllWindows()

#%%

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label: num for num , label in enumerate(actions)}
label_map

sequences , labels = [],[]

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
        

#%%

X = np.array(sequences)
X.shape

y = to_categorical(labels).astype(int)
y

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.05)
X_train.shape

#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64,return_sequences=True,activation= 'relu',input_shape = (30,1662)))
model.add(LSTM(128,return_sequences=True,activation= 'relu'))
model.add(LSTM(64,return_sequences=False,activation= 'relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax'))


#%%
actions.shape[0]

res = [.7,0.2,0.1]
#%%
model.compile(optimizer='Adam', loss= 'categorical_crossentropy',metrics=['categorical_accuracy'])
#%%
model.fit(X_train,y_train,epochs=2000,callbacks=[tb_callback])

#%%

res = model.predict(X_test)
res[0]
np.sum(res[0]) 

actions[np.argmax(res[1])]
actions[np.argmax(y_test[1])]

model.save('actions.h5')
model.load_weights('actions.h5')

#%%

from sklearn.metrics import multilabel_confusion_matrix , accuracy_score

yhat = model.predict(X_train)

ytrue = np.argmax(y_train,axis=1).tolist()
yhat = np.argmax(yhat,axis=1).tolist()

multilabel_confusion_matrix(ytrue,yhat)
accuracy_score(ytrue, yhat)

#%%

sequence = []
sentence = []
threshold = 0.6
predictions = []



cap = cv2.VideoCapture(0) # Bilgisayarınızın varsayılan kamerasını aktif hale getirir (0 genellikle varsayılan kamerayı işaret eder).
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
   while cap.isOpened(): # Kamera açık olduğu sürece döngü devam eder.
        ret , frame = cap.read() # Kameradan bir kare (frame) yakalar. ret, kare yakalamanın başarılı olup olmadığını (True/False) belirten bir boolean değerdir.
        
        image , results1 = mediapipe_detection(frame,holistic) # Yani, image değişkeni işlenmiş frame değerini alır ve results değişkeni holistic modelinin işlem sonucunu alır
        
        draw_landmarks(image, results1) # işlenmiş frame i ve sonuçları draw_landmarks foksiyonuna göndeririz
        
        
        keypoints = extract_keypoint(results1)
        sequence.append(keypoints)
        sequence = sequence[-10:]
       
        if len(sequence) == 10:
           res = model.predict(np.expand_dims(sequence, axis=0))[0]
           print(actions[np.argmax(res)])
           predictions.append(np.argmax(res))
           
           
        #3. Viz logic
           if np.unique(predictions[-10:])[0]==np.argmax(res): 
               if res[np.argmax(res)] > threshold: 
                   
                   if len(sentence) > 0: 
                       if actions[np.argmax(res)] != sentence[-1]:
                           sentence.append(actions[np.argmax(res)])
                   else:
                       sentence.append(actions[np.argmax(res)])

           if len(sentence) > 5: 
               sentence = sentence[-5:]

           
           
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        cv2.imshow('OpenCV Feed', image) #Yakalanan kareyi bir pencerede gösterir. Pencere başlığı 'OpenCV Feed' olarak ayarlandı.
        #cv2.imshow fonksiyonu, OpenCV kütüphanesinin bir parçasıdır ve bu fonksiyon, bir görüntüyü belirli bir pencere üzerinde göstermek için kullanılır.
        if cv2.waitKey(10) & 0xFF == ord('q'): # kamera q ya basınca kapanır
            #Her 10 milisaniyede bir, klavyeden bir tuşa basılıp basılmadığını kontrol eder. Eğer 'q' tuşuna basılırsa, döngüyü kırarak görüntü yakalama işlemini sonlandırır.
            '''
            & İşareti:
            & işareti, bit düzeyinde bir AND işlemidir. Yani, iki sayının bitleri karşılaştırılır ve her iki sayının ilgili bitleri de 1 ise, sonuçtaki bit 1 olur; aksi takdirde 0 olur. Bu işlem, belirli bit maskelerini uygulamak için kullanılır.
            0xFF:
            0xFF hexadecimal sistemde 255 demektir ve binary olarak 11111111 olarak temsil edilir. Bu, 8 bitlik bir maske anlamına gelir.
            cv2.waitKey(10) fonksiyonu, belirtilen süre (milisaniyeler cinsinden) boyunca bir klavye tuşuna basılıp basılmadığını kontrol eder ve basılan tuşun ASCII değerini bir tam sayı olarak döndürür.
            Fakat cv2.waitKey 32 bitlik bir değer döndürür ve genellikle bu değerin son 8 biti tuş bilgisini içerir. & 0xFF işlemi, döndürülen değerin sadece son 8 bitini (yani gerçek tuş değerini) elde etmek için kullanılır.
            İfade Anlamı:
            cv2.waitKey(10) & 0xFF, cv2.waitKey tarafından döndürülen değerin son 8 bitini maskeleyerek, gerçek tuş değerini çıkarır.
            ord('q') ise, 'q' karakterinin ASCII değerini verir. ASCII tablosunda 'q' harfinin değeri 113'tür.
            Dolayısıyla, if cv2.waitKey(10) & 0xFF == ord('q') ifadesi, eğer kullanıcı son 10 milisaniye içinde 'q' tuşuna basmışsa True değerini döndürür ve bu koşul sağlandığında döngü kırılır (program kamera görüntüsünü göstermeyi durdurur).
            '''
            break
   cap.release() # cv2.VideoCapture tarafından başlatılan video yakalama cihazını (örneğin, bir kamera) serbest bırakır. Bu fonksiyon çağrıldığında, OpenCV video yakalama cihazı ile olan bağlantıyı keser. Böylece, kamera gibi donanım kaynakları başka uygulamalar tarafından kullanılabilir hale gelir.
   cv2.destroyAllWindows()# OpenCV ile açılan tüm pencereyi kapatır. Eğer programınız bir veya birden fazla pencere oluşturduysa (örneğin, cv2.imshow() fonksiyonu ile görüntü gösterimi yapılıyorsa), bu fonksiyon tüm bu pencereleri kapatır ve kaynakları temizler.
   