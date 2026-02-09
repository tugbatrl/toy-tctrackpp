#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray 
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np
import sys
import os
import time
from ultralytics import YOLO as yolo

# --- DOSYA YOLLARI ---

# Dosyanın olduğu yeri otomatik bulur
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from pysot.core.config import cfg
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
from pysot.tracker.tctrackplus_tracker import TCTrackplusTracker
from pysot.utils.model_load import load_pretrain

class TakipciDugumu(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.get_logger().info("TCTrack++ Takip Düğümü Başlatılıyor...")

        # 1. ARAÇLARI HAZIRLA
        self.bridge = CvBridge()
        self.tracker = None
        self.takip_modu = False 
        
        # 2. AYARLAR
        self.KAYIP_ESIGI = 0.80      # Biraz düşürdüm ki hemen pes etmesin
        self.KILITLENME_SURESI = 4.0 # 4 Saniye kuralı
        self.TOLERANS_SURESI = 1.0   # 1 Saniye tolerans
        
        # SAYAÇLAR
        self.dark_frame_count = 0

        self.locked_start = None     # Kilitlenme başlangıcı
        self.last_seen_time = 0 # Son görülme zaman
        self.basarili_sure = 0.0 
        self.harcanan_tolerans = 0.0
        self.last_loop_time = time.time() # Döngü süresi hesabı için

        # 3. MODELLERİ YÜKLE
        self.init_tctrack()
        self.init_yolo()

        # 4. YAYINCILAR VE ABONE


        self.bbox_pub = self.create_publisher(Float32MultiArray, '/tracker/bbox', 10)
        self.debug_pub = self.create_publisher(Image, '/tracker/debug_image', 10)

        topic_name = '/world/default/model/rc_cessna_mono_cam_0/link/camera_link/sensor/camera/image'

        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.resim_geldi_callback,
            10
        )
        
        self.get_logger().info("Sistem Hazır kamera görüntüsü bekleniyor...")

    def init_yolo(self):
        try:
            # Model yolunu garantiye alalım
            yolo_path = os.path.join(current_dir, "snapshot", "best.pt")
                
            self.detector = yolo(yolo_path)
            self.get_logger().info("YOLO Hazır!")
        except Exception as e:
            self.get_logger().error(f"YOLO Hatası: {e}")

    def init_tctrack(self):
        try:
            config_path = os.path.join(current_dir, 'experiments', 'TCTrack', 'config.yaml')
            snapshot_path = os.path.join(current_dir, 'snapshot', 'TCTrack.pth')

            cfg.merge_from_file(config_path)
            cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
            device = torch.device('cuda' if cfg.CUDA else 'cpu')

            model = ModelBuilder_tctrackplus('test')
            model = load_pretrain(model, snapshot_path).eval().to(device)
            self.tracker = TCTrackplusTracker(model)
            self.hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

            self.get_logger().info(f"TCTrack++ Yüklendi! Cihaz: {device}")
        except Exception as e:
            self.get_logger().error(f"TCTrack Hatası: {e}")

    def resim_geldi_callback(self, msg):

        self.this_time = time.time()

        # --- ROS MESAJINI OPENCV GÖRÜNTÜSÜNE ÇEVİR ---
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            return

        # --- ZAMANI GÜNCELLE (EN ÖNEMLİ KISIM) ---
        dt = self.this_time - self.last_loop_time
        self.last_loop_time = self.this_time

        # Koordinatları al
                 
        h_img, w_img, _ = cv_image.shape
        #sarı kutu sınırları               
        x_baslangic = int(w_img * 0.25)
        x_bitis = int(w_img * 0.75)
        y_baslangic = int(h_img * 0.10)
        y_bitis = int(h_img * 0.90)

        cv2.rectangle(cv_image , (x_baslangic , y_baslangic) ,(x_bitis , y_bitis), (0,255,255) , 2)

        # ---------------------------------------------------------
        # DURUM 1: ARAMA MODU (YOLO)
        # ---------------------------------------------------------
        if not self.takip_modu:
            results = self.detector(cv_image, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    
                    if conf > 0.60: # %60'tan eminse

                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords)
                        
                        w = x2 - x1
                        h = y2 - y1

                        # AŞIRI BÜYÜK HEDEF KONTROLÜ (Ekranın %90'ı ise alma)
                        if w > w_img * 0.9: continue
                        

                        ucak_center_x = int(x1 + (w/2))
                        ucak_center_y = int(y1 + (h/2))

                        cv2.circle(cv_image , (ucak_center_x , ucak_center_y) , 5 , (0,0,255), -1)

                        ucak_buyuk_mu= (h >= (h_img * 0.07)) and ((w >= (w_img * 0.07)))
                        ucak_icerde_mi= (x_baslangic < ucak_center_x  < x_bitis) and (y_baslangic < ucak_center_y < y_bitis)

                        if not ucak_buyuk_mu or not ucak_icerde_mi: continue


                        # TCTrack Başlat
                        init_rect = (x1, y1, w, h)
                        self.tracker.init(cv_image, init_rect)
                        
                        # DURUMLARI GÜNCELLE
                        self.takip_modu = True
                        self.locked_start = self.this_time   # Kronometre Başladı!
                        self.last_seen_time = self.this_time # Şimdi gördüm!
                        
                        self.get_logger().info("HEDEF BULUNDU! Sayaç Başlıyor...")
                        self.harcanan_tolerans = 0.0
                        
                        # İlk kareyi hemen çiz (Kullanıcı görsün)
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(cv_image, "Takip ediliyor", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0 ), 2)
                        break 
                if self.takip_modu: break

        # ---------------------------------------------------------
        # DURUM 2: TAKİP MODU (TCTrack++)
        # ---------------------------------------------------------
        else:
            
            outputs = self.tracker.track(cv_image, self.hp)
            score = outputs['best_score']
            bbox = list(map(int, outputs['bbox']))
            
            # bu kısım eğer olur da yolo yanlış şeyleri parametre olarak gönderirse takip algoritması kafayı yemesin diye önlem kısmı
            # eğer kapkaranlıksa  takip etmeyi bırakacağız
            x, y, w, h = bbox
            img_h, img_w, _ = cv_image.shape
            
            # Sınır Kontrolü
            x = max(0, min(x, img_w-1)); y = max(0, min(y, img_h-1))
            w = max(1, min(w, img_w - x)); h = max(1, min(h, img_h - y))
            
            roi = cv_image[y:y+h, x:x+w]
            
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                mean_val = np.mean(gray_roi)

                # Karanlık Testi
                if mean_val < 10: 
                    self.dark_frame_count += 1
                    if self.dark_frame_count >= 5:
                        self.takip_modu = False; self.dark_frame_count = 0
                        self.get_logger().warn("ORTAM KARANLIK - Takip Bitti"); return
                else: self.dark_frame_count = 0


            # ---------------------------------------------------------
            # KRONOMETRE VE TOLERANS MANTIĞI (FİNAL VERSİYON)
            # ---------------------------------------------------------
            
            # 1. HEDEFİ GÖRÜYORSAK (SKOR İYİ - TCTrack Takip Ediyor)
            if score >= self.KAYIP_ESIGI:
                
                # Tolerans sayacını sıfırla çünkü hedefi görüyoruz
                self.harcanan_tolerans = 0.0
                self.last_seen_time = self.this_time 

                # Görüntü Boyutları
                h_img, w_img, _ = cv_image.shape # Bunu buradan alalım garanti olsun

            
                # Hedef Analizi
                ucak_center_x = int(x + (w/2))
                ucak_center_y = int(y + (h/2))

                # Şartlar
                ucak_buyuk_mu = (h >= (h_img * 0.07)) and (w >= (w_img * 0.07))
                ucak_icerde_mi = (x_baslangic < ucak_center_x < x_bitis) and (y_baslangic < ucak_center_y < y_bitis)

                # --- SENARYO A: ŞARTLAR UYGUN (KİLİTLENME SÜRECİ) ---
                if ucak_icerde_mi and ucak_buyuk_mu:
                    
                    # Eğer sayaç daha önce başlamadıysa başlat
                    if self.locked_start is None:
                        self.locked_start = self.this_time

                    # Geçen Süreyi Hesapla
                    gecen_sure = self.this_time - self.locked_start
                    
                    # Ekrana Yaz (Geri Sayım)
                    cv2.putText(cv_image, f"LOCKING: {gecen_sure:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # ÇİZİM - YEŞİL KUTU (Şartlar Tamam)
                    p1 = (bbox[0], bbox[1])
                    p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                    cv2.rectangle(cv_image, p1, p2, (0, 255, 0), 3) # Yeşil
                    cv2.putText(cv_image, f"TRACKING ({score:.2f})", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 4 SANİYE KONTROLÜ (ZAFER ANI!!!)
                    if gecen_sure >= self.KILITLENME_SURESI:
                        cv2.putText(cv_image, "LOCKED SUCCESSFULLY", (bbox[0], bbox[1]-40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3) # Yeşil Yazı
                        self.get_logger().info("KİLİTLENME BAŞARILI!")
                
                # --- SENARYO B: TAKİP VAR AMA ŞARTLAR YOK (DIŞARIDA VEYA KÜÇÜK) ---
                else:
                    # Kural ihlali var, süreyi SIFIRLA!
                    self.locked_start = None 
                    
                    # Uyarı Yazısı
                    if not ucak_buyuk_mu:
                        durum_mesaji = "Hedefe yaklas"
                    elif not ucak_icerde_mi:
                        durum_mesaji = "Hedef merkez dışında"

                        
                    cv2.putText(cv_image, durum_mesaji, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # ÇİZİM - KIRMIZI KUTU (Takip var ama kilit yok)
                    p1 = (bbox[0], bbox[1])
                    p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                    cv2.rectangle(cv_image, p1, p2, (0, 0, 255), 2) # Kırmızı
                    cv2.putText(cv_image, f"TRACKING - NO LOCK", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


            # 2. HEDEF KAYIPSA (SKOR DÜŞÜK - TCTrack Kaçırdı)
            else:
                # (Senin yazdığın kısım aynen kalıyor, orası doğruydu)
                self.harcanan_tolerans += dt # Buradaki dt yukarıda hesaplanmalı
                
                if self.harcanan_tolerans < self.TOLERANS_SURESI:
                    cv2.putText(cv_image, "KAYIP - BEKLENIYOR...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    # Çizim yapmıyoruz çünkü kutu yok
                else:
                    self.takip_modu = False
                    self.locked_start = None
                    self.harcanan_tolerans = 0.0
                    return

        # --- GÖRÜNTÜYÜ GÖSTER VE YAYINLA ---
        # 1. ROS Topic olarak bas
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.debug_pub.publish(debug_msg)
        except Exception:
            pass

        # 2. Ekrana Pencere Aç (Window)
        cv2.namedWindow("Takip Sistemi", cv2.WINDOW_NORMAL)
        cv2.imshow("Takip Sistemi", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TakipciDugumu()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()