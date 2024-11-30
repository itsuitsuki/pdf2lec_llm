import React, { useState, useEffect, useRef } from "react";
import * as pdfjs from "pdfjs-dist";
import "pdfjs-dist/build/pdf.worker.entry";
import "../styles/Display.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faFilePdf,
  faChevronDown,
  faChevronUp,
} from "@fortawesome/free-solid-svg-icons";
import Transcript from "./Transcript";
import { getAuthHeaders } from "../utils/auth";
import { getAudioBlob } from "../utils/audio";

interface CarouselProps {
  pdfId: string;
  audioTimestamps: number[];
  timestamp: string;
}

interface SlideViewerProps {
  slides: string[];
  currentSlide: number;
}

const SlideViewer: React.FC<SlideViewerProps> = ({ slides, currentSlide }) => {
  console.log("Rendering slide", currentSlide); // Debugging log to track slide change

  return (
    <div className="slide-viewer">
      {slides.length > 0 && (
        <img
          src={slides[currentSlide]}
          alt={`Slide ${currentSlide + 1}`}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            border: "1px solid gray",
          }}
        />
      )}
    </div>
  );
};

const Carousel: React.FC<CarouselProps> = ({ pdfId, audioTimestamps, timestamp }) => {
  const [slides, setSlides] = useState<string[]>([]);
  const [currentSlide, setCurrentSlide] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [loading, setLoading] = useState(true);
  const [audioUrl, setAudioUrl] = useState<string>('');
  const baseAudioUrl = `http://localhost:5000/data/${pdfId}/generated_audios/combined.mp3`;

  // 修改 URL 构建方式，添加认证头
  const headers = getAuthHeaders();
  const pdfUrl = `http://localhost:5000/data/${pdfId}/${timestamp}`;

  useEffect(() => {
    const loadPdf = async () => {
      try {
        console.log('Loading PDF from URL:', pdfUrl);
        const token = localStorage.getItem('token');
        const loadingTask = pdfjs.getDocument({
          url: pdfUrl,
          httpHeaders: {
            'Authorization': `Bearer ${token}`
          }
        });
        const pdf = await loadingTask.promise;
        console.log('PDF loaded successfully, pages:', pdf.numPages);
        const slideImages: string[] = [];

        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const viewport = page.getViewport({ scale: 1 });
          const canvas = document.createElement("canvas");
          const context = canvas.getContext("2d");
          canvas.width = viewport.width;
          canvas.height = viewport.height;

          await page.render({ canvasContext: context!, viewport }).promise;
          slideImages.push(canvas.toDataURL());
        }

        setSlides(slideImages);
      } catch (error) {
        console.error('Error loading PDF:', error);
        console.error('PDF URL was:', pdfUrl);
      } finally {
        setLoading(false);
      }
    };

    loadPdf();
  }, [pdfUrl]);

  useEffect(() => {
    const loadAudio = async () => {
      try {
        const url = await getAudioBlob(baseAudioUrl);
        setAudioUrl(url);
      } catch (error) {
        console.error('Error loading audio:', error);
      }
    };
    
    loadAudio();
    
    return () => {
      // Clean up object URL when component unmounts
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [baseAudioUrl]);

  useEffect(() => {
    const audioElement = audioRef.current;
    if (!audioElement || slides.length === 0) return;

    const handleTimeUpdate = () => {
      const currentTime = audioElement.currentTime * 1000; // 转换为毫秒
      
      // 根据时间戳确定当前应该显示的幻灯片
      const newSlideIndex = audioTimestamps.findIndex((timestamp, index) => {
        const nextTimestamp = audioTimestamps[index + 1];
        return currentTime >= timestamp && (!nextTimestamp || currentTime < nextTimestamp);
      });

      if (newSlideIndex !== -1 && newSlideIndex !== currentSlide) {
        setCurrentSlide(newSlideIndex);
      }
    };

    audioElement.addEventListener("timeupdate", handleTimeUpdate);
    return () => {
      audioElement.removeEventListener("timeupdate", handleTimeUpdate);
    };
  }, [currentSlide, slides, audioTimestamps]);

  if (loading) {
    return <div>Loading slides...</div>;
  }

  const handleThumbnailClick = (index: number) => {
    setCurrentSlide(index);
    if (audioRef.current && audioTimestamps[index]) {
      audioRef.current.currentTime = audioTimestamps[index] / 1000; // 转换回秒
    }
  };

  const getVisibleThumbnails = () => {
    const totalSlidesToShow = 4;
    const start = Math.max(0, Math.min(currentSlide - 2));
    const end = Math.min(slides.length, start + totalSlidesToShow);
    return slides.slice(start, end);
  };

  return (
    <div className="carousel-container">
      <div className="hover-container">
        <SlideViewer slides={slides} currentSlide={currentSlide} />

        <audio ref={audioRef} className="audio-player" controls>
          <source src={audioUrl} type="audio/mpeg" />
          Your browser does not support the audio element.
        </audio>
      </div>
      <div style={{display:"flex", flexDirection:"column", width:"90%", flexGrow:"1"}}>
        <div
          className="collapsed-hover-container"
          style={{
            display: "flex",
            flexDirection: "column",
            position: "relative",
          }}
        >
          <div
            className={`thumbnail-container ${collapsed ? "collapsed" : ""}`}
          >
            {getVisibleThumbnails().map((thumbnail, index) => {
              const slideIndex = Math.max(0, currentSlide - 2) + index;
              return (
                <img
                  key={slideIndex}
                  src={thumbnail}
                  alt={`Thumbnail ${slideIndex + 1}`}
                  onClick={() => handleThumbnailClick(slideIndex)}
                  style={{
                    width: "22%",
                    height: "100%",
                    cursor: "pointer",
                    border:
                      slideIndex === currentSlide
                        ? "2px solid #757bf5"
                        : "1px solid #c3c3c3",
                  }}
                />
              );
            })}
          </div>
          <button
            className="centered-button"
            onClick={() => {
              console.log("Button clicked!");
              setCollapsed(!collapsed);
            }}
          >
            <div className="icon-container">
              {collapsed ? (
                <FontAwesomeIcon icon={faChevronUp} />
              ) : (
                <FontAwesomeIcon icon={faChevronDown} />
              )}
              <FontAwesomeIcon icon={faFilePdf} />
            </div>
          </button>
          
        </div>
        <Transcript />
      </div>
    </div>
  );
};

export default Carousel;
