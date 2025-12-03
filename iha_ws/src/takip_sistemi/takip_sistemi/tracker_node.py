#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray # Koordinatları (x,y,w,h) yollamak için
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np
import sys
import os
#şimdilik yoloyu burdan çağırıyorum
from ultralytics import YOLO as yolo

# --- TCTRACK++ KÜTÜPHANELERİNİ YÜKLEME ---
# Kodumuzun bulunduğu klasörü buluyoruz
current_dir = '/home/tugba/Toy-iha/iha_ws/src/takip_sistemi/takip_sistemi'
sys.path.append(current_dir) # Bu klasörü Python'un arama yoluna ekle

# Artık yanımızdaki klasörleri sanki kütüphaneymiş gibi çağırabiliriz
from pysot.core.config import cfg
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
from pysot.tracker.tctrackplus_tracker import TCTrackplusTracker
from pysot.utils.model_load import load_pretrain

class TakipciDugumu(Node):
    def __init__(self):
        super().__init__('tracker_node') # Düğümün adı
        self.get_logger().info("TCTrack++ Takip Düğümü Başlatılıyor...")

        # 1. ARAÇLARI HAZIRLA
        self.bridge = CvBridge()
        self.tracker = None
        
        # Takip Durumu (State)
        self.takip_modu = False # Henüz takip etmiyoruz
        
        # 2. MODELİ YÜKLE (Aşağıdaki fonksiyonu çağırıyoruz)
        self.init_tctrack()
        self.init_yolo()

        #kayıp eşiği ekliyorum takip doğruluk oranı buranın altına düşerse takibi bırakıp yoloyu çalıştıracak
        self.KAYIP_ESIGI = 0.90
        self.dark_frame_count = 0
        self.flat_frame_count = 0

         #şimdilik yoloyu kendim entegre ediyorum
        
        

        # 3. YAYINCILAR (PUBLISHERS - AĞIZ) 
        # Pilot için koordinatları yayınla
        self.bbox_pub = self.create_publisher(Float32MultiArray, '/tracker/bbox', 10)
        # Hakem için boyanmış resmi yayınla
        self.debug_pub = self.create_publisher(Image, '/tracker/debug_image', 10)

        # 4. ABONE (SUBSCRIBER - KULAK) 
        # Kameradan gelen resmi dinle
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw', # Dinleyeceğimiz konu
            self.resim_geldi_callback,
            10)
        
        self.get_logger().info("Hazırım! Kamera verisi bekleniyor... ")

    def init_yolo(self):
        try:
            yolo_path = os.path.join(current_dir, "snapshot/best.pt")
            self.detector = yolo(yolo_path)
            self.get_logger().info("dedektif hazır 1")
        except Exception as e:
            self.get_logger().error(e)
    

    def init_tctrack(self):
        """ TCTrack++ modelini ve ayarlarını yükler """
        try:
            # Dosya yollarını oluştur
            
            # current_dir yerine doğrudan senin klasörünün adresini yazıyoruz.
        
            # Dosyalarımın olduğu ana klasör
            base_path = '/home/tugba/Toy-iha/iha_ws/src/takip_sistemi/takip_sistemi'
        
            config_path = os.path.join(base_path, 'experiments', 'TCTrack', 'config.yaml')
            snapshot_path = os.path.join(base_path, 'snapshot', 'TCTrack.pth')


            # Ayarları yükle
            cfg.merge_from_file(config_path)
            
            # Ekran kartı (CUDA) var mı?
            cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
            device = torch.device('cuda' if cfg.CUDA else 'cpu')

            # Modeli inşa et
            model = ModelBuilder_tctrackplus('test')
            # Ağırlıkları yükle
            model = load_pretrain(model, snapshot_path).eval().to(device)

            # Takipçiyi oluştur
            self.tracker = TCTrackplusTracker(model)

            self.hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

            self.get_logger().info(f" Model Yüklendi! Cihaz: {device}")
            
        except Exception as e:
            self.get_logger().error(f" Model yüklenirken hata: {e}")

    def resim_geldi_callback(self, msg):
        """ Her yeni resim geldiğinde burası çalışır """
        try:
            # 1. ROS Resmini -> OpenCV Resmine çevir
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Resim çeviri hatası: {e}")
            return

        if not self.takip_modu:

            results = self.detector(cv_image , verbose = False) #verbose false logları kirletmesin diye

            for result in results:
                boxes = result.boxes
                for box in boxes: #eğer modelin bulduğu oran %60 tan büyükse (yani ihaya %60 tan fazla benziyorsa takip edelim)
                    conf = float(box.conf[0])
                    if conf >0.60:
                        print(conf)
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords)

                        w = x2 - x1
                        h = y2 - y1
                        init_rect = (x1,y1,w,h)

                        # img_h, img_w, _ = cv_image.shape
                        # if w > img_w * 0.8 or h > img_h * 0.8:
                        #     self.get_logger().warn(" ÇOK BÜYÜK HEDEF! Yoksayılıyor...")
                        #     continue

                        

                        self.tracker.init(cv_image , init_rect)
                        self.takip_modu = True
                        self.get_logger().info("iha hedefi bulundu. Takip başlatılıyor")
                        break

        else:
            # Takip devam ediyor

            

            outputs = self.tracker.track(cv_image, self.hp)
            score = outputs['best_score']
            bbox = list(map(int, outputs['bbox']))
            
            # bu kısım eğer olur da yolo yanlış şeyleri parametre olarak gönderirse takip algoritması kafayı yemesin diye önlem kısmı 
            # eğer kapkaranlıksa ve ya dümdüz duvarı takip ediyorsa takip etmeyi bırakacağız 
            x, y, w, h = bbox
            
            # Görüntü sınırlarını taşmamak için önlem (Clip)
            img_h, img_w, _ = cv_image.shape
            x = max(0, min(x, img_w))
            y = max(0, min(y, img_h))
            w = max(1, min(w, img_w - x)) # Genişlik en az 1 olsun
            h = max(1, min(h, img_h - y)) # Yükseklik en az 1 olsun
            
            # 1. Kutunun içindeki resmi kesip al (ROI - Region of Interest)
            roi = cv_image[y:y+h, x:x+w]
            
            # 2. İstatistikleri hesapla
            # Griye çevir (Tek kanalda işlem yapmak kolaydır)
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Ortalama Parlaklık (Mean) ve Varyans (StdDev)
                mean_val = np.mean(gray_roi)
                std_dev = np.std(gray_roi)
                

                # 3. KARANLIK TESTİ (Kamerayı kapatınca)
                if mean_val < 10: # Simsiyahsa (Eşik: 10)
                    self.get_logger().warn(" görüntü karanlık. Takip Bırakıldı.")
                    self.dark_frame_count += 1
                    if self.dark_frame_count >= 5 :
                        self.dark_frame_count = 0
                        self.takip_modu = False
                        return
                else:
                    self.dark_frame_count = 0

                # 4. DÜZLÜK TESTİ (Duvar/Boşluk)
                if std_dev < 10: # Görüntüde hiç detay yoksa, dümdüz renksizse
                    
                    self.get_logger().warn("düz şeyler takip ediliyor. Takip Bırakıldı.")
                    self.flat_frame_count +=1
                    if self.flat_frame_count >= 5 :
                        self.flat_frame_count = 0
                        self.takip_modu = False
                        return
                else:
                    self.dark_frame_count = 0
            # ---------------------------------------------

            # Eğer buraya kadar geldiyse, görüntü sağlamdır.
            # Skor kontrolünü de yapalım
            if score < self.KAYIP_ESIGI:
                self.takip_modu = False
                self.get_logger().warn(f"İha kayboldu tekrar aranıyor")
                return

            # ... (Buradan sonrası çizim ve yayınlama kodları) ...

            # 3. ÇİZİM YAP (Hakem için) 
            # Kutu çiz (Kırmızı: 0, 0, 255)
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            cv2.rectangle(cv_image, p1, p2, (0, 0, 255), 3)
            
            # Yazı yaz
            cv2.putText(cv_image, "LOCKED", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # 4. SONUÇLARI YAYINLA (ROS'a geri ver) 
            
            # A) Koordinatları Yolla
            bbox_msg = Float32MultiArray()
            bbox_msg.data = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            self.bbox_pub.publish(bbox_msg)

            # B) Boyanmış Resmi Yolla
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TakipciDugumu()
    rclpy.spin(node) # Sonsuz döngü (Düğüm kapanana kadar)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()