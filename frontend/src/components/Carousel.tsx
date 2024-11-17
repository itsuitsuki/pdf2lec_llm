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
interface CarouselProps {
  pdfUrl: string;
  audioUrl: string;
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

const Carousel: React.FC<CarouselProps> = ({ pdfUrl, audioUrl }) => {
  const [slides, setSlides] = useState<string[]>([]);
  const [currentSlide, setCurrentSlide] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  // An array that stores the time in seconds for each slide.
  const slideTimes: number[] = [];
  for (let i = 0; i < slides.length; i++) {
    slideTimes.push(i * 2); // Increment by 5 for each slide
  }
  useEffect(() => {
    const loadPdf = async () => {
      const loadingTask = pdfjs.getDocument(pdfUrl);
      const pdf = await loadingTask.promise;
      const slideImages: string[] = [];

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const viewport = page.getViewport({ scale: 1 });

        const scale = 1;
        const scaledViewport = page.getViewport({ scale });

        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");
        canvas.width = scaledViewport.width;
        canvas.height = scaledViewport.height;

        await page.render({ canvasContext: context!, viewport: scaledViewport })
          .promise;
        slideImages.push(canvas.toDataURL());
      }

      setSlides(slideImages);
    };

    loadPdf();
  }, [pdfUrl]);

  useEffect(() => {
    const audioElement = audioRef.current;
    if (!audioElement || slides.length === 0) return;

    const handleTimeUpdate = () => {
      const currentTime = audioElement.currentTime;

      // Determine the correct slide based on the current time in the audio
      let newSlide = currentSlide;
      for (let i = 0; i < slideTimes.length; i++) {
        if (
          currentTime >= slideTimes[i] &&
          (i === slideTimes.length - 1 || currentTime < slideTimes[i + 1])
        ) {
          newSlide = i;
          break;
        }
      }

      // Only update `currentSlide` if it’s different to avoid unnecessary re-renders
      if (newSlide !== currentSlide) {
        setCurrentSlide(newSlide);
      }
    };

    // Attach the time update listener
    audioElement.addEventListener("timeupdate", handleTimeUpdate);

    // Cleanup listener on unmount
    return () => {
      audioElement.removeEventListener("timeupdate", handleTimeUpdate);
    };
  }, [currentSlide, slides, slideTimes]);

  if (slides.length === 0) {
    return <div>Loading...</div>;
  }

  const handleThumbnailClick = (index: number) => {
    if (currentSlide < index && currentSlide < slides.length - 1) {
      setCurrentSlide(currentSlide + 1);
    } else if (currentSlide > index && currentSlide > 0) {
      setCurrentSlide(currentSlide - 1);
    } else if (currentSlide === index) {
      setCurrentSlide(index); // In case the user clicks on the current slide itself
    }
    if (audioRef.current) {
      audioRef.current.currentTime = slideTimes[index];
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
