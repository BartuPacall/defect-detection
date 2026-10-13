\# Çelik Yüzey Kusur Tespiti (NEU-DET)



NEU-DET veri seti kullanılarak çelik yüzeylerdeki 6 farklı kusur tipini (`crazing`, `inclusion`, `patches`, `pitted\_surface`, `rolled-in\_scale`, `scratches`) tespit eden ve konumlandıran bir nesne tespiti projesi.



\## Kurulum



```bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

```



`neu\_det.yaml` dosyasındaki `path` satırı yerel makineye özgü bir yol içerir. Projeyi klonlayanların bu yolu kendi `yolo\_data` klasörlerinin konumuna göre güncellemesi gerekir.



`yolo\_data/` klasörü, Pascal VOC XML anotasyonlarını YOLO formatına dönüştüren aşağıdaki komutla oluşturulur:



```bash

python src/convert\_to\_yolo.py

```



\## YOLO ile Kusur Tespiti ve Konumlandırma



\*\*Veri seti:\*\* 1440 eğitim ve 360 doğrulama görseli (200×200 piksel), 6 sınıf.



\*\*Donanım:\*\* RTX 3050 Laptop GPU (4 GB VRAM).



\### Model Karşılaştırması



| Model | imgsz | Parametre | Precision | Recall | mAP50 | mAP50-95 |

|---|---:|---:|---:|---:|---:|---:|

| YOLO11n | 320 | 2,6 M | 0.589 | 0.713 | 0.716 | 0.375 |

| YOLO11s | 320 | 9,4 M | 0.673 | 0.638 | 0.695 | 0.372 |

| YOLO11s | 640 | 9,4 M | 0.673 | 0.721 | 0.750 | 0.394 |

| YOLO11n | 640 | 2,6 M | 0.720 | 0.687 | 0.751 | 0.405 |



\*\*Seçilen model:\*\* YOLO11n (`imgsz=640`). YOLO11s ile mAP50 açısından eşit, mAP50-95'te biraz daha yüksek ve 3,6 kat daha az parametreye sahip. Çıkarım süresi yaklaşık \*\*3,5 ms/görsel\*\* olarak ölçüldü.



!\[Tahmin örneği](reports/yolo11n\_640/val\_batch0\_pred.jpg)



\### Bulgular



\- \*\*Çözünürlük model boyutundan daha etkili oldu.\*\* Aynı çözünürlükte (320) daha büyük model daha iyi sonuç vermedi.

\- \*\*En zayıf sınıflar:\*\* `crazing` (mAP50: 0.567) ve `rolled-in\_scale` (mAP50: 0.534). Bu sınıflar üç farklı ayarda da düşük performans gösterdi. Kusur sınırlarının belirsiz olması olası nedenlerden biri.

\- \*\*Yanlış pozitifler önemli bir sorun.\*\* Baseline modelin (YOLO11n, imgsz=320) karışıklık matrisinde (güven eşiği 0.25), özellikle `crazing` ve `rolled-in\_scale` sınıflarında etiketlenmemiş bölgelerin kusur olarak tahmin edildiği görüldü.



\### Sınırlılıklar



\- Her model ayarı yalnızca bir kez çalıştırıldı. Bu nedenle 0.751 ve 0.750 mAP50 sonuçları arasında anlamlı bir fark olduğu söylenemez.

\- Model seçimi ve performans raporlaması aynı doğrulama seti üzerinden yapıldı. Bağımsız bir test seti bulunmadığından sonuçlar gerçek performansı bir miktar yüksek gösterebilir.

