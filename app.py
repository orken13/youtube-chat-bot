# chatbot_refactored.py

import os
import streamlit as st
import tempfile
import uuid
from io import BytesIO
import time
import requests
from decouple import config
import minio
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import yt_dlp
import shutil



def tema_ayarla():
    st.markdown("""
    <style>
    /* Ana arka plan */
    .stApp {
        background: linear-gradient(to bottom right, #2D033B, #000000);
        color: #E5B8F4;
    }

    /* Başlık */
    .stTitleBlock {
        color: #E5B8F4;
    }

    /* Metin alanları */
    .stTextInput > div > div > input {
        background-color: #3F0071;
        color: #E5B8F4;
        border: 1px solid #810CA8;
    }

    /* Metin alanı */
    .stTextArea > div > div > textarea {
        background-color: #3F0071;
        color: #E5B8F4;
        border: 1px solid #810CA8;
    }

    /* Butonlar */
    .stButton > button {
        background-color: #810CA8;
        color: #E5B8F4;
        border: none;
    }

    .stButton > button:hover {
        background-color: #C147E9;
        color: white;
    }

    /* Sohbet mesajı konteyneri */
    .stChatMessage {
        background-color: #00000;
        border: 1px solid #810CA8;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }

    /* Kullanıcı mesajı */
    .stChatMessage.user {
        background-color: #810CA8;
    }

    /* Sohbet giriş alanı */
    .stChatInput {
        border-color: #810CA8;
    }

    /* Sohbet giriş alanı fokus */
    .stChatInput:focus {
        border-color: #C147E9;
    }

    /* Başlıklar */
    h1, h2, h3, h4, h5, h6 {
        color: #E5B8F4;
    }

    /* Metin */
    p, li, span {
        color: #E5B8F4;
    }
    </style>
    """, unsafe_allow_html=True)


def minio_kurulum():
    try:
        istemci = minio.Minio(
            endpoint=config("MINIO_ENDPOINT", default="localhost:9000"),
            access_key=config("MINIO_ACCESS_KEY", default="minioadmin"),
            secret_key=config("MINIO_SECRET_KEY", default="minioadmin"),
            secure=config("MINIO_SECURE", default="False").lower() == "true"
        )

        bucket_ismi = config("MINIO_BUCKET", default="youtube-videos-14")
        if not istemci.bucket_exists(bucket_ismi):
            istemci.make_bucket(bucket_ismi)

        return istemci, bucket_ismi
    except Exception as e:
        st.error(f"MinIO kurulum hatası: {e}")
        return None, None


def youtube_video_indir_ve_isle(url):
    try:
        gecici_klasor = tempfile.mkdtemp()
        st.info("YouTube videosu indiriliyor...")

        # Tamamen güvenli UUID tabanlı bir dosya adı kullan
        dosya_uuid = str(uuid.uuid4())
        video_ydl_secenekleri = {
            'format': 'best',  # En iyi video kalitesi
            'outtmpl': os.path.join(gecici_klasor, f'{dosya_uuid}.%(ext)s'),  # UUID tabanlı dosya adı
            'quiet': True,
            'ignoreerrors': True,  # Hataları görmezden gel ve devam et
        }

        with yt_dlp.YoutubeDL(video_ydl_secenekleri) as ydl:
            bilgi = ydl.extract_info(url, download=True)
            if not bilgi:
                raise Exception("Video bilgisi alınamadı.")

            video_basligi = bilgi.get('title', 'Bilinmeyen Video')
            video_dosyasi = ydl.prepare_filename(bilgi)

            # Dosya var mı kontrol et
            if not os.path.exists(video_dosyasi):
                raise Exception(f"Video dosyası indirilemedi: {video_dosyasi}")

            st.info(f"Video indirildi: {video_basligi}. Şimdi MinIO'ya yükleniyor...")

        istemci, bucket_ismi = minio_kurulum()
        # Daha önce oluşturulan UUID'yi kullan
        video_id = dosya_uuid

        if not istemci:
            raise Exception("MinIO depolama alanına bağlanılamadı")

        # Video dosyasını MinIO'ya yükle
        try:
            with open(video_dosyasi, 'rb') as dosya_verisi:
                dosya_stat = os.stat(video_dosyasi)
                video_uzantisi = os.path.splitext(video_dosyasi)[1]
                istemci.put_object(
                    bucket_name=bucket_ismi,
                    object_name=f"{video_id}/video{video_uzantisi}",
                    data=dosya_verisi,
                    length=dosya_stat.st_size,
                    content_type=f"video/{video_uzantisi[1:]}"  # .mp4 -> video/mp4
                )
        except Exception as e:
            raise Exception(f"Video MinIO'ya yüklenirken hata: {e}")

        st.info(f"Video MinIO'ya yüklendi. Şimdi ses ayıklanıyor...")

        # UUID tabanlı ses dosyası adı kullan
        ses_dosyasi = os.path.join(gecici_klasor, f"{dosya_uuid}_audio.mp3")

        # ffmpeg ile ses dönüştürme
        try:
            import subprocess
            ffmpeg_komut = [
                'ffmpeg',
                '-i', video_dosyasi,
                '-q:a', '0',  # En yüksek ses kalitesi
                '-map', 'a',  # Sadece ses kanalını al
                '-vn',  # Video kanalını kaldır
                ses_dosyasi
            ]

            subprocess.run(ffmpeg_komut, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Ses dosyasının oluşup oluşmadığını kontrol et
            if not os.path.exists(ses_dosyasi) or os.path.getsize(ses_dosyasi) == 0:
                raise Exception("Ses dosyası oluşturulamadı.")

        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg hatası: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        except Exception as e:
            raise Exception(f"Ses dönüştürme hatası: {e}")

        st.info(f"Ses başarıyla ayıklandı ve MP3'e dönüştürüldü. MinIO'ya yükleniyor...")

        # Ses dosyasını MinIO'ya yükle
        try:
            with open(ses_dosyasi, 'rb') as dosya_verisi:
                dosya_stat = os.stat(ses_dosyasi)
                istemci.put_object(
                    bucket_name=bucket_ismi,
                    object_name=f"{video_id}/ses.mp3",
                    data=dosya_verisi,
                    length=dosya_stat.st_size,
                    content_type="audio/mp3"
                )
        except Exception as e:
            raise Exception(f"Ses dosyası MinIO'ya yüklenirken hata: {e}")

        minio_url = f"http://{config('MINIO_ENDPOINT', default='localhost:9000')}/{bucket_ismi}/{video_id}/ses.mp3"

        st.success(f"İşlem başarılı: {video_basligi}")
        return minio_url, video_id, ses_dosyasi, video_basligi, gecici_klasor

    except Exception as e:
        st.error(f"Video indirme ve işleme hatası: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None

def ses_transkript_et(ses_dosyasi):
    try:
        st.info("Ses dosyası transkript ediliyor...")

        import openai
        istemci = openai.OpenAI(api_key=config("OPENAI_API_KEY"))

        with open(ses_dosyasi, "rb") as ses:
            transkripsiyon = istemci.audio.transcriptions.create(
                model="whisper-1",
                file=ses
            )

        transkript = transkripsiyon.text
        return transkript

    except Exception as e:
        st.error(f"Transkript hatası: {e}")
        return None


def vektor_db_olustur(transkript):
    try:
        st.info("Vektör veritabanı oluşturuluyor...")


        metin_bolme = RecursiveCharacterTextSplitter(
            chunk_size=1500,# 1500 karakterlik parçalara böl
            chunk_overlap=150#Üst üste 150 karakter koy
        )

        parcalar = metin_bolme.split_text(transkript)


        embedding_fonksiyonu = OpenAIEmbeddings(model="text-embedding-3-small")
        vektor_db = FAISS.from_texts(
            parcalar,
            embedding_fonksiyonu
        )

        return vektor_db

    except Exception as e:
        st.error(f"Vektör veritabanı oluşturma hatası: {e}")
        return None


def video_isle(url):
    try:
        ilerleme_cubugu = st.progress(0)


        os.environ['OPENAI_API_KEY'] = config("OPENAI_API_KEY")


        minio_url, video_id, ses_dosyasi, baslik, gecici_klasor = youtube_video_indir_ve_isle(url)
        if not minio_url or not video_id:
            raise Exception("YouTube videosu indirilemedi, MinIO'ya kaydedilemedi veya ses ayıklanamadı")
        ilerleme_cubugu.progress(40)


        transkript = ses_transkript_et(ses_dosyasi)
        if not transkript:
            raise Exception("Ses transkript edilemedi")
        ilerleme_cubugu.progress(70)


        vektor_db = vektor_db_olustur(transkript)
        if not vektor_db:
            raise Exception("Vektör veritabanı oluşturulamadı")
        ilerleme_cubugu.progress(90)


        if gecici_klasor and os.path.exists(gecici_klasor):
            shutil.rmtree(gecici_klasor)
            st.info("Geçici dosyalar temizlendi.")

        ilerleme_cubugu.progress(100)
        return vektor_db, video_id

    except Exception as e:
        st.error(f"Video işleme hatası: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def soru_cevap_zinciri_olustur(vektor_db):
    sablon = """Bir videonun bağlamını verin. Kullanıcıya samimi ve kesin bir şekilde cevap verin.

    Bağlam: {context}

    İnsan: {question}

    AI:"""

    prompt = PromptTemplate(
        template=sablon,
        input_variables=["context", "question"]
    )

    qa_zinciri = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        ),
        chain_type="stuff",
        retriever=vektor_db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_zinciri



def main():
    tema_ayarla()

    st.title("🎬 YouTube Video Chatbot")
    st.subheader("YouTube video içeriğine dayalı soru-cevap sistemi")


    if "islendi" not in st.session_state:
        st.session_state.islendi = False

    if "mesajlar" not in st.session_state:
        st.session_state.mesajlar = []

    if "vektor_db" not in st.session_state:
        st.session_state.vektor_db = None

    if "video_id" not in st.session_state:
        st.session_state.video_id = None


    youtube_url = st.text_input("YouTube Video URL'si", "")

    islem_butonu = st.button("Videoyu İşle 🚀")

    if islem_butonu:
        if not youtube_url:
            st.error("Lütfen bir YouTube URL'si girin.")
        else:
            st.session_state.vektor_db, st.session_state.video_id = video_isle(youtube_url)

            if st.session_state.vektor_db:
                st.session_state.islendi = True
                st.session_state.mesajlar = [
                    {"role": "assistant",
                     "content": "Merhaba! YouTube videosu işlendi. Video hakkında bana sorular sorabilirsiniz."}
                ]

    st.markdown("---")


    if st.session_state.islendi and st.session_state.vektor_db:

        for mesaj in st.session_state.mesajlar:
            with st.chat_message(mesaj["role"]):
                st.write(mesaj["content"])


        kullanici_girisi = st.chat_input("Videonun içeriği hakkında bir soru sorun...")

        if kullanici_girisi:

            st.session_state.mesajlar.append({"role": "user", "content": kullanici_girisi})


            with st.chat_message("user"):
                st.write(kullanici_girisi)


            with st.chat_message("assistant"):
                with st.spinner("Yanıt hazırlanıyor..."):

                    qa_zinciri = soru_cevap_zinciri_olustur(st.session_state.vektor_db)


                    yanit = qa_zinciri({"query": kullanici_girisi})
                    st.write(yanit["result"])


            st.session_state.mesajlar.append({"role": "assistant", "content": yanit["result"]})
    else:
        st.info("👆 Lütfen bir YouTube videosu URL'si girin ve işleyin.")


if __name__ == "__main__":
    main()