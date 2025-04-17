import React, { useState, useEffect, useRef, useCallback } from "react";

const SpeakPopup = ({ onClose, setQuery }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [transcript, setTranscript] = useState("");
    const [error, setError] = useState("");
    const [isSpeechRecognitionSupported, setIsSpeechRecognitionSupported] = useState(true);
    const recognitionRef = useRef(null);
    const timeoutRef = useRef(null);

    // Create a memoized stopRecording function that won't change between renders
    const stopRecording = useCallback(() => {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);

        try {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
            setIsRecording(false);
        } catch (err) {
            console.error("Error stopping speech recognition:", err);
        }
    }, []);

    // Handle silence detection with a ref to track isRecording state
    // This prevents closure issues with the callback
    const isRecordingRef = useRef(isRecording);
    
    // Keep the ref in sync with state
    useEffect(() => {
        isRecordingRef.current = isRecording;
    }, [isRecording]);

    // Initialize speech recognition on component mount
    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setIsSpeechRecognitionSupported(false);
            setError("Speech recognition is not supported in this browser.");
        } else {
            // Initialize speech recognition
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = true;
            recognitionRef.current.interimResults = true;
            recognitionRef.current.lang = 'en-US';

            // Setup event handlers
            recognitionRef.current.onresult = (event) => {
                let finalTranscript = '';
                let interimTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                    } else {
                        interimTranscript += transcript;
                    }
                }

                // Display transcript
                setTranscript(finalTranscript || interimTranscript);

                // Reset auto-stop timer
                if (timeoutRef.current) clearTimeout(timeoutRef.current);
                timeoutRef.current = setTimeout(() => {
                    // Use the ref instead of the closed-over variable
                    if (isRecordingRef.current) stopRecording();
                }, 3000); // Auto-stop after 3 seconds of silence
            };

            recognitionRef.current.onerror = (event) => {
                console.error("Speech recognition error:", event.error);
                setError(`Error: ${event.error}`);
                setIsRecording(false);
            };

            recognitionRef.current.onend = () => {
                setIsRecording(false);
            };
        }

        // Cleanup
        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.onresult = null;
                recognitionRef.current.onerror = null;
                recognitionRef.current.onend = null;
            }
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
        };
    }, [stopRecording]); // Now depends on stopRecording which is memoized

    // Handle 30-second fallback timeout when recording starts
    useEffect(() => {
        if (!isRecording) return;
        
        const fallbackTimeout = setTimeout(() => {
            stopRecording();
        }, 30000);
        
        return () => clearTimeout(fallbackTimeout);
    }, [isRecording, stopRecording]);

    const startRecording = () => {
        setError("");
        setTranscript("");
        try {
            recognitionRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Error starting speech recognition:", err);
            setError(`Failed to start recording: ${err.message}`);
        }
    };

    const handleUseTranscript = () => {
        if (transcript.trim()) {
            setQuery(transcript);
            onClose();
        }
    };

    return (
        <div className="popup-content">
            <h2>Voice Input</h2>

            {!isSpeechRecognitionSupported ? (
                <div className="voice-input-error">
                    <p>{error}</p>
                    <p>Try using a browser like Chrome, Edge, or Safari that supports the Web Speech API.</p>
                </div>
            ) : (
                <>
                    <div className="voice-input-container">
                        <div className={`voice-icon ${isRecording ? 'recording' : ''}`}>
                            {isRecording ? '🔴' : '🎤'}
                        </div>

                        {isRecording ? (
                            <p>Listening... Speak now</p>
                        ) : (
                            <p>{transcript ? "Recording complete!" : "Click to start speaking..."}</p>
                        )}

                        {error && <p className="voice-error">{error}</p>}
                    </div>

                    {transcript && (
                        <div className="transcript-container">
                            <h3>Transcript:</h3>
                            <p className="transcript-text">{transcript}</p>
                        </div>
                    )}

                    <div className="popup-actions">
                        {isRecording ? (
                            <button
                                onClick={stopRecording}
                                className="recording-btn"
                            >
                                Stop Recording
                            </button>
                        ) : (
                            <>
                                <button
                                    onClick={startRecording}
                                    className="primary-btn"
                                    disabled={!isSpeechRecognitionSupported}
                                >
                                    {transcript ? "Record Again" : "Start Recording"}
                                </button>

                                {transcript && (
                                    <button
                                        onClick={handleUseTranscript}
                                        className="primary-btn use-transcript-btn"
                                    >
                                        Use This Text
                                    </button>
                                )}
                            </>
                        )}
                        <button onClick={onClose} className="secondary-btn">Cancel</button>
                    </div>
                </>
            )}
        </div>
    );
};

export default SpeakPopup;