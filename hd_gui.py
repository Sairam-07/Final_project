import tkinter as tk
from tkinter import filedialog
import subprocess




def run_live_hd_program():
    import cv2
    import numpy as np

    # Function to dehaze an image using the Dark Channel Prior algorithm
    def dark_channel (image, window_size=15):
        "Calculate the dark channel of an image."
        min_channel= np.min(image, axis=2)
        return cv2.erode (min_channel, np. ones ((window_size, window_size)))
    def estimate_atmosphere (image, dark_channel, percentile=0.001):
        """Estimate the atmosphere light of the image"""
        flat_dark_channel= dark_channel.flatten()
        flat_image = image.reshape (-1, 3)
        num_pixels = flat_image.shape [0]
        num_pixels_to_keep = int(num_pixels*percentile)
        indices = np.argpartition (flat_dark_channel, -num_pixels_to_keep) [-num_pixels_to_keep:]
        atmosphere = np.max (flat_image [indices], axis=0)
        return atmosphere
    def dehaze(image, tmin=0.1, omega=0.95, window_size=15):
        """Dehaze the input image using the Dark Channel Prior algorithm."""
        if image is None:
                return None
        image = image.astype (np.float64) / 255.0
        dark_ch = dark_channel (image, window_size)
        atmosphere = estimate_atmosphere(image, dark_ch)
        transmission = 1 - omega * dark_ch
        transmission = np.maximum (transmission, tmin)
        dehazed = np. zeros_like (image)
        for channel in range (3):
            dehazed [:, :, channel] = (image[:, :, channel]-atmosphere [channel]) / transmission + atmosphere [channel]
        dehazed=np.clip (dehazed, 0, 1)
        dehazed =(dehazed * 255).astype (np.uint8)
        return dehazed

    def detect_humans(frame):
            net = cv2.dnn.readNetFromCaffe(r"C:\Users\HP\Desktop\FinalProjectFolderWithReferences[1]\FinalProjectFolder\CodeDumpDonotTouch\MobileNetSSD_deploy.prototxt", r"c:\Users\HP\Desktop\FinalProjectFolderWithReferences[1]\FinalProjectFolder\CodeDumpDonotTouch\MobileNetSSD_deploy.caffemodel")
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            humans = []
            frame_height, frame_width = frame.shape[:2]

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Filter out weak detections
                    class_id = int(detections[0, 0, i, 1])
                    if class_id == 15:  # Class ID for person in MobileNet SSD
                        box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                        (startX, startY, endX, endY) = box.astype("int")
                        humans.append((startX, startY, endX - startX, endY - startY))

            return humans

    def dehaze_and_detect_humans( output_video_path):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open input video file.")
            return
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        net = cv2.dnn.readNetFromCaffe(r"C:\Users\HP\Desktop\FinalProjectFolderWithReferences[1]\FinalProjectFolder\CodeDumpDonotTouch\MobileNetSSD_deploy.prototxt", r"c:\Users\HP\Desktop\FinalProjectFolderWithReferences[1]\FinalProjectFolder\CodeDumpDonotTouch\MobileNetSSD_deploy.caffemodel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            dehazed_frame = dehaze(frame)
            
            blob = cv2.dnn.blobFromImage(cv2.resize(dehazed_frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            frame_height, frame_width = dehazed_frame.shape[:2]

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Filter out weak detections
                    class_id = int(detections[0, 0, i, 1])
                    if class_id == 15:  # Class ID for person in MobileNet SSD
                        box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(dehazed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            cv2.imshow('Dehazed and Detected Humans', dehazed_frame)
            out.write(dehazed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if __name__ == "__main__":
	#give path to your project folder here
        output_video_path = r"C:\Users\hp\Downloads\output_video2.mp4"
        dehaze_and_detect_humans(output_video_path)


def run_video_hd_program():
        file_path = filedialog.askopenfilename()
        if file_path:
            import cv2
            import numpy as np

            def dark_channel (image, window_size=15):
                "Calculate the dark channel of an image."
                min_channel= np.min(image, axis=2)
                return cv2.erode (min_channel, np. ones ((window_size, window_size)))
            def estimate_atmosphere (image, dark_channel, percentile=0.001):
                """Estimate the atmosphere light of the image"""
                flat_dark_channel= dark_channel.flatten()
                flat_image = image.reshape (-1, 3)
                num_pixels = flat_image.shape [0]
                num_pixels_to_keep = int(num_pixels*percentile)
                indices = np.argpartition (flat_dark_channel, -num_pixels_to_keep) [-num_pixels_to_keep:]
                atmosphere = np.max (flat_image [indices], axis=0)
                return atmosphere
            def dehaze(image, tmin=0.1, omega=0.95, window_size=15):
                """Dehaze the input image using the Dark Channel Prior algorithm."""
                if image is None:
                    return None
                image = image.astype (np.float64) / 255.0
                dark_ch = dark_channel (image, window_size)
                atmosphere = estimate_atmosphere(image, dark_ch)
                transmission = 1 - omega * dark_ch
                transmission = np.maximum (transmission, tmin)
                dehazed = np. zeros_like (image)
                for channel in range (3):
                    dehazed [:, :, channel] = (image[:, :, channel]-atmosphere [channel]) / transmission + atmosphere [channel]
                dehazed=np.clip (dehazed, 0, 1)
                dehazed =(dehazed * 255).astype (np.uint8)
                return dehazed

            def detect_humans(frame):
                net = cv2.dnn.readNetFromCaffe(r"C:\Users\HP\Desktop\FinalProjectFolderWithReferences[1]\FinalProjectFolder\CodeDumpDonotTouch\MobileNetSSD_deploy.prototxt", r"c:\Users\HP\Desktop\FinalProjectFolderWithReferences[1]\FinalProjectFolder\CodeDumpDonotTouch\MobileNetSSD_deploy.caffemodel")
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                humans = []
                frame_height, frame_width = frame.shape[:2]

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # Filter out weak detections
                        class_id = int(detections[0, 0, i, 1])
                        if class_id == 15:  # Class ID for person in MobileNet SSD
                            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                            (startX, startY, endX, endY) = box.astype("int")
                            humans.append((startX, startY, endX - startX, endY - startY))

                return humans




            # Main function to dehaze the video and detect humans
            def dehaze_and_detect_humans(input_video_path, output_video_path):
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    print("Error: Unable to open input video file.")
                    return
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                
                net = cv2.dnn.readNetFromCaffe(r"C:\Users\HP\Desktop\project_code\MobileNetSSD_deploy.prototxt", r"C:\Users\HP\Desktop\project_code\MobileNetSSD_deploy.caffemodel")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    dehazed_frame = dehaze(frame)
                    
                    blob = cv2.dnn.blobFromImage(cv2.resize(dehazed_frame, (300, 300)), 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    detections = net.forward()

                    frame_height, frame_width = dehazed_frame.shape[:2]

                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5:  # Filter out weak detections
                            class_id = int(detections[0, 0, i, 1])
                            if class_id == 15:  # Class ID for person in MobileNet SSD
                                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                                (startX, startY, endX, endY) = box.astype("int")
                                cv2.rectangle(dehazed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    cv2.imshow('Dehazed and Detected Humans', dehazed_frame)
                    out.write(dehazed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                out.release()
                cv2.destroyAllWindows()
            if __name__ == "__main__":
                input_video_path = file_path
		#give path to your project folder
                output_video_path = r"C:\Users\hp\Downloads\output_video1.mp4"
                dehaze_and_detect_humans(input_video_path, output_video_path)



def run_image_hd_program():
        file_path = filedialog.askopenfilename()
        if file_path:
            import cv2
            import numpy as np
            def dark_channel (image, window_size=15):
                "Calculate the dark channel of an image."
                min_channel= np.min(image, axis=2)
                return cv2.erode (min_channel, np. ones ((window_size, window_size)))
            def estimate_atmosphere (image, dark_channel, percentile=0.001):
                """Estimate the atmosphere light of the image"""
                flat_dark_channel= dark_channel.flatten()
                flat_image = image.reshape (-1, 3)
                num_pixels = flat_image.shape [0]
                num_pixels_to_keep = int(num_pixels*percentile)
                indices = np.argpartition (flat_dark_channel, -num_pixels_to_keep) [-num_pixels_to_keep:]
                atmosphere = np.max (flat_image [indices], axis=0)
                return atmosphere
            def dehaze(image, tmin=0.1, omega=0.95, window_size=15):
                """Dehaze the input image using the Dark Channel Prior algorithm."""
                if image is None:
                    return None
                image = image.astype (np.float64) / 255.0
                dark_ch = dark_channel (image, window_size)
                atmosphere = estimate_atmosphere(image, dark_ch)
                transmission = 1 - omega * dark_ch
                transmission = np.maximum (transmission, tmin)
                dehazed = np. zeros_like (image)
                for channel in range (3):
                    dehazed [:, :, channel] = (image[:, :, channel]-atmosphere [channel]) / transmission + atmosphere [channel]
                dehazed=np.clip (dehazed, 0, 1)
                dehazed =(dehazed * 255).astype (np.uint8)
                return dehazed

            def detect_humans(frame):
                net = cv2.dnn.readNetFromCaffe(r"C:\Users\HP\Desktop\project_code\MobileNetSSD_deploy.prototxt", r"C:\Users\HP\Desktop\project_code\MobileNetSSD_deploy.caffemodel")
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                humans = []
                frame_height, frame_width = frame.shape[:2]

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # Filter out weak detections
                        class_id = int(detections[0, 0, i, 1])
                        if class_id == 15:  # Class ID for person in MobileNet SSD
                            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                            (startX, startY, endX, endY) = box.astype("int")
                            humans.append((startX, startY, endX - startX, endY - startY))

                return humans

            def dehaze_and_detect_humans(input_image_path):
                """Load the image"""
                image = cv2.imread(input_image_path)
                if image is None:
                    print("Error: Could not load the input image.")
                    return

                """Apply dehazing"""  
                dehazed_image = dehaze(image)
                
                """Detect humans in the dehazed image"""
                detected_humans = detect_humans(dehazed_image)

                """Draw rectangles around detected humans"""
                for (startX, startY, width, height) in detected_humans:
                    cv2.rectangle(dehazed_image, (startX, startY), (startX + width, startY + height), (0, 255, 0), 2)

                """Display and save the dehazed image"""
                cv2.imshow('Dehazed and Detected Humans', dehazed_image)
                cv2.imwrite('output.jpg', dehazed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if __name__ == "__main__":
                input_image_path = file_path
                dehaze_and_detect_humans(input_image_path)
                                                                                                       

root = tk.Tk()
root.geometry("600x250")
root.title("Human Detection and Dehazing")

button_image_hd=tk.Button(root, text="Dehaze and human detection in image", command=run_image_hd_program)
button_image_hd.pack(pady=10)

button_live_hd=tk.Button(root, text="Dehaze and human detection in Live", command=run_live_hd_program)
button_live_hd.pack(pady=10)

button_video_hd=tk.Button(root, text="Dehaze and human detection in Video", command=run_video_hd_program)
button_video_hd.pack(pady=10)

message_label = tk.Label(root, text="Press 'g' to quit", fg="red")
message_label.pack(pady=20)

root.mainloop()