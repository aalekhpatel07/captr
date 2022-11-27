use captrs::*;
use tokio::{signal::unix::{signal, SignalKind}, select, join};
use std::{sync::{Arc, Mutex}, path::PathBuf, str::FromStr};
use image::*;
use std::time::Duration;
use tracing::{
    info,
    warn,
    debug,
    error,
    trace
};
use tokio::sync::mpsc::unbounded_channel;
use thiserror::Error;
use clap::{Parser, arg};
use chrono::{DateTime, Utc};
use std::time::SystemTime;


pub fn now_iso() -> String {
    let now: DateTime<Utc> = SystemTime::now().into();
    now.to_rfc3339()
}


#[derive(PartialEq)]
pub enum RecordingState
{
    Stopped,
    Started
}


#[derive(Error, Debug)]
pub enum CaptrError {
    #[error("Some thing inner: {0}")]
    FrameSend(String),
    #[error("Inner Captrs: {0}")]
    InnerCapturer(String),
    #[error("Something IO {0}")]
    IO(String),
    #[error("H264: {0}")]
    EncodeH264(String)
}


impl From<std::io::Error> for CaptrError {
    fn from(err: std::io::Error) -> Self {
        Self::IO(err.to_string())
    }
}


pub type Frame = ImageBuffer<Rgb<u8>, Vec<u8>>;

pub struct FrameCapture {
    capturer: Capturer,
    state: RecordingState,
    frame_rate: f32
}

unsafe impl Send for FrameCapture {}
unsafe impl Sync for FrameCapture {}


impl FrameCapture {

    pub fn new(capturer: Capturer, recording_state: RecordingState, frame_rate: f32) -> Self {
        Self { capturer, state: recording_state, frame_rate }
    }

    pub fn start(&mut self) -> Result<(), CaptrError> {
        self.state = RecordingState::Started;
        debug!("Started recording at {:#?} fps.", self.frame_rate);
        Ok(())
    }
    pub fn stop(&mut self) -> Result<(), CaptrError> {
        self.state = RecordingState::Stopped;
        Ok(())
    }

    pub fn capture(&mut self) -> Result<Frame, CaptrError> {
        let (width, height) = self.capturer.geometry();

        let captured_frame = self.capturer.capture_frame();
        match captured_frame {
            Ok(frame) => {
                let mut rgb = RgbImage::new(width, height);
                let mut all_pixels = frame.into_iter();
                
                for x in 0..height {
                    for y in 0..width {
                        let Bgr8 { r, g, b, ..} = all_pixels.next().unwrap();
                        *rgb.get_pixel_mut(y, x) = [r, g, b].into();
                    }
                }
                Ok(rgb)
            },
            Err(err) => {
                Err(CaptrError::InnerCapturer(format!("{:#?}", err)))
            }
        }
    }

    pub fn step(&mut self) -> Result<Option<Frame>, CaptrError> {
        match self.state {
            RecordingState::Started => {
                Ok(Some(self.capture()?))
            },
            RecordingState::Stopped => Ok(None)
        }
    }

}

mod video {
    use std::{io::{Cursor, Read, Seek, SeekFrom}, path::Path};
    use image::EncodableLayout;
    use minimp4::Mp4Muxer;
    use openh264::encoder::{Encoder, EncoderConfig};
    use tokio::sync::mpsc::UnboundedReceiver;
    use tracing::{info, debug};

    use crate::Frame;

    pub struct Video<P> {
        encoder: Encoder,
        buffer: Vec<u8>,
        width: u32,
        height: u32,
        name: String,
        output_path: P
    }

    unsafe impl<P> Send for Video<P> {}
    unsafe impl<P> Sync for Video<P> {}

    impl<P: AsRef<Path>> Video<P> {
        pub fn with_config(
            width: u32, 
            height: u32, 
            name: &str,
            output_path: P,
        ) -> Self {
            Self {
                encoder: Encoder::with_config(EncoderConfig::new(width, height)).unwrap(),
                buffer: vec![],
                width,
                height,
                name: name.to_owned(),
                output_path
            }
        }

        pub fn write_frame(&mut self, frame: Frame) -> Result<usize, crate::CaptrError> {
            let mut yuv = openh264::formats::RBGYUVConverter::new(self.width as usize, self.height as usize);
            yuv.convert(frame.as_bytes());

            match self.encoder.encode(&yuv) {
                Ok(bitstream) => {
                    let frame_size = bitstream.raw_info().iFrameSizeInBytes;
                    bitstream.write_vec(&mut self.buffer);
                    Ok(frame_size as usize)
                },
                Err(err) => {
                    Err(crate::CaptrError::EncodeH264(err.to_string()))
                }
            }
        }
        
        fn convert_mp4(&mut self) -> Cursor<Vec<u8>> {
            let mut video_buffer = Cursor::new(Vec::new());
            let mut mp4muxer = Mp4Muxer::new(&mut video_buffer);
            mp4muxer.init_video(
                i32::try_from(self.width).unwrap(), 
                i32::try_from(self.height).unwrap(), 
                false,
                &self.name
            );
            mp4muxer.write_video(&self.buffer);
            mp4muxer.close();

            video_buffer
        }

        pub fn save(&mut self) -> Result<usize, crate::CaptrError> {
            let mut video_buffer = self.convert_mp4();
            video_buffer.seek(SeekFrom::Start(0))?;
            let mut video_bytes = Vec::new();
            video_buffer.read_to_end(&mut video_bytes)?;
            let total_bytes_written = video_bytes.len();
            std::fs::write(&self.output_path, &video_bytes)?;
            debug!("Written {} bytes to {:#?}", total_bytes_written, &self.output_path.as_ref());
            info!("Screen capture saved to {:#?}", &self.output_path.as_ref());
            Ok(total_bytes_written)
        }

        pub fn clear(&mut self) -> Result<(), crate::CaptrError> {
            self.buffer.clear();
            Ok(())
        }

    }

}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 60.0, help="The frame rate for screen capture.")]
    fps: f32
}


#[tokio::main]
async fn main() -> Result<(), CaptrError> {

    let args = Args::parse();

    tracing_subscriber::fmt::init();

    let capturer = match Capturer::new(0) {
        Ok(capturer) => capturer,
        Err(err) => {
            return Err(CaptrError::InnerCapturer(err.to_string()));
        }
    };

    let video_name = now_iso();
    let output_path = format!("ScreenCapture_{}.mp4", now_iso());
    let (width, height) = capturer.geometry();
    let output_path = PathBuf::from_str(&output_path).unwrap();

    let mut signal_stream = signal(SignalKind::hangup())?;

    let (frame_tx, mut frame_rx) = unbounded_channel();


    let video_writer = Arc::new(Mutex::new(video::Video::with_config(
        width, 
        height, 
        &video_name, 
        output_path,
    )));

    let frame_capture = Arc::new(Mutex::new(
        FrameCapture::new(capturer, RecordingState::Stopped, args.fps)
    ));

    let frame_capture_cp = frame_capture.clone();
    let video_writer_cp = video_writer.clone();

    let (_tx, mut _rx) = unbounded_channel();


    let duration = Duration::from_millis((1_000.0 / args.fps).round() as u64);

    let t0 = tokio::spawn(async move {
        loop {
            _tx.send(()).unwrap();
            tokio::time::sleep(duration).await;
        }
    });

    let t1 = tokio::spawn(async move {
        loop {
            select! {
                _ = signal_stream.recv() => {
                    let mut recording_cp = frame_capture_cp.lock().unwrap();
                    let mut video_writer_guard = video_writer_cp.lock().unwrap();

                    match recording_cp.state {
                        RecordingState::Started => {
                            recording_cp.stop().unwrap();
                            // video_writer_guard.
                            info!("Stopping recording.");
                            video_writer_guard.save().unwrap();
                        },
                        RecordingState::Stopped => {
                            recording_cp.start().unwrap();
                            video_writer_guard.clear().unwrap();

                            info!("Starting recording.");
                        }
                    }
                },
                _ = _rx.recv() => {
                    let mut recording_cp = frame_capture_cp.lock().unwrap();
                    if let Some(frame) = recording_cp.step().unwrap() {
                        frame_tx.send(frame).unwrap();
                    }
                }
            }
        }
    });

    let t2 = tokio::spawn(async move {
        loop {
            let frame = frame_rx.recv().await;
            let mut video_writer = video_writer.lock().unwrap();
            if let Some(frame) = frame {
                video_writer.write_frame(frame).unwrap();
            }
        }
    });

    let _ = join!(t0, t1, t2);
    Ok(())

}
