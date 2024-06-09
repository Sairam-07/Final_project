import tkinter as tk
from tkinter import filedialog
import subprocess

def run_image_program():
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
        if __name__ == "__main__":
            """Specify the path to your image file"""
            image_path=file_path
            # Attempt to load the image.
            input_image=cv2.imread (image_path)
            # Check if the image was loaded successfully
            if input_image is not None:
                """Apply dehazing"""  
                output_image = dehaze (input_image)
                if output_image is not None:
                    """Save the dehazed image with the desired output path and filename."""
                    cv2.imwrite('output.jpg', output_image)
                    """Display the dehazed image"""
                    """Display the dehazed image"""
                    cv2.imshow ('Dehazed Image', output_image)
                    cv2.waitKey (0)
                else:
                    print("Error: Failed to dehaze the image.")
            else:
                """ The image loading failed"""
                print ("Error: Could not load the input image.")

def run_video_program():
    file_path = filedialog.askopenfilename()
    if file_path:
        import cv2
        import numpy as np
        def dark_channel(image, window_size=15):
            """Calculate the dark channel of an image."""
            min_channel= np.min(image, axis=2)
            return cv2.erode (min_channel, np.ones ((window_size, window_size)))
        def estimate_atmosphere (image, dark_channel, percentile=0.001):
            """Estimate the atmosphere light of the image."""
            flat_dark_channel=dark_channel.flatten ()
            flat_image=image. reshape (-1, 3)
            num_pixels=flat_image.shape [0]
            num_pixels_to_keep=int (num_pixels*percentile)
            indices = np.argpartition (flat_dark_channel, -num_pixels_to_keep) [-num_pixels_to_keep:]
            atmosphere = np.max (flat_image [indices], axis=0)
            return atmosphere
        def dehaze (image, tmin=0.1, omega=0.95, window_size=15):
            "Dehaze the input image using the Dark Channel Prior algorithm."""
            if image is None:
                return None
            image=image.astype (np. float64) / 255.0
            dark_ch=dark_channel (image, window_size)
            atmosphere = estimate_atmosphere (image, dark_ch)
            transmission = 1- omega*dark_ch
            transmission = np.maximum (transmission, tmin)
            dehazed=np. zeros_like (image)
            for channel in range (3):
                dehazed [:,:, channel]=(image[:, :, channel]-atmosphere [channel]) / transmission + atmosphere [channel]
            dehazed=np.clip (dehazed, 0, 1)
            dehazed=(dehazed*255).astype (np. uint8)
            return dehazed
        if __name__ =="__main__":
            #Specify the path to the video file
            video_path=file_path
            #Open the video file
            cap = cv2.VideoCapture (video_path)
            # Get the width and height of the video frames
            frame_width = int (cap.get (cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int (cap.get (cv2.CAP_PROP_FRAME_HEIGHT))
            # Create a VideoWriter object to save the dehazed video
            fource = cv2.VideoWriter_fourcc(*'mp4v')
            #give path to your project folder
            output_video_path=r""
            output_video=cv2.VideoWriter (output_video_path, fource, 25.0, (frame_width, frame_height))
            while True:
            #Read the next frame from the video file
                ret, frame = cap.read()
                #If the frame is not empty, dehaze it and write it to the output video
                if ret:
                    dehazed_frame=dehaze (frame)
                    output_video.write(dehazed_frame)
                    # Display the dehazed frame.
                    cv2.imshow ('Dehazed Video', dehazed_frame)
                    #Press 'q' to quit.
                    if cv2.waitKey (1) & 0xFF == ord ('g'):
                        break
                # If the frame is empty, break the loop
                else:
                    break
        # Release the video capture and video writer objects
            cap.release()
            output_video.release()
            cv2.destroyAllWindows()

def run_live_program():
    import cv2
    import numpy as np
    def dark_channel(image, window_size=15):
        """Calculate the dark channel of an image."""
        min_channel= np.min(image, axis=2)
        return cv2.erode (min_channel, np.ones ((window_size, window_size)))
    def estimate_atmosphere (image, dark_channel, percentile=0.001):
        """Estimate the atmosphere light of the image."""
        flat_dark_channel=dark_channel.flatten ()
        flat_image=image. reshape (-1, 3)
        num_pixels=flat_image.shape [0]
        num_pixels_to_keep=int (num_pixels*percentile)
        indices = np.argpartition (flat_dark_channel, -num_pixels_to_keep) [-num_pixels_to_keep:]
        atmosphere = np.max (flat_image [indices], axis=0)
        return atmosphere
    def dehaze (image, tmin=0.1, omega=0.95, window_size=15):
        "Dehaze the input image using the Dark Channel Prior algorithm."""
        if image is None:
            return None
        image=image.astype (np. float64) / 255.0
        dark_ch=dark_channel (image, window_size)
        atmosphere = estimate_atmosphere (image, dark_ch)
        transmission = 1- omega*dark_ch
        transmission = np.maximum (transmission, tmin)
        dehazed=np. zeros_like (image)
        for channel in range (3):
            dehazed [:,:, channel]=(image[:, :, channel]-atmosphere [channel]) / transmission + atmosphere [channel]
        dehazed=np.clip (dehazed, 0, 1)
        dehazed=(dehazed*255).astype (np. uint8)
        return dehazed
    if __name__ =="__main__":
        #Specify the path to the video file
        cap = cv2.VideoCapture(0)
        # Get the width and height of the video frames
        frame_width = int (cap.get (cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int (cap.get (cv2.CAP_PROP_FRAME_HEIGHT))
        # Create a VideoWriter object to save the dehazed video
        fource = cv2.VideoWriter_fourcc(*'mp4v')
	#give path to your project folder
        output_video_path=r""
        output_video=cv2.VideoWriter (output_video_path, fource, 25.0, (frame_width, frame_height))
        while True:
        #Read the next frame from the video file
            ret, frame = cap.read()
            #If the frame is not empty, dehaze it and write it to the output video
            if ret:
                dehazed_frame=dehaze (frame)
                output_video.write(dehazed_frame)
                # Display the dehazed frame.
                cv2.imshow ('Dehazed Video', dehazed_frame)
                #Press 'q' to quit.
                #write a line to press q or g to stop the program
                if cv2.waitKey (1) & 0xFF == ord ('g'):
                    break
            # If the frame is empty, break the loop
            else:
                break
    # Release the video capture and video writer objects
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()



root = tk.Tk()
root.geometry("600x250")
root.title("Only Dehaze")

button_image = tk.Button(root, text="Dehaze Image", command=run_image_program)
button_image.pack(pady=10)

button_video = tk.Button(root, text="Dehaze video", command=run_video_program)
button_video.pack(pady=10)

button_live=tk.Button(root, text="Dehaze in Live", command=run_live_program)
button_live.pack(pady=10)

message_label = tk.Label(root, text="Press 'g' to quit", fg="red")
message_label.pack(pady=20)

root.mainloop()