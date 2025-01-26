from app.VideoPlayer import VideoPlayer
import customtkinter


def main ():
    
    # Initialise the root widget.
    root_widget = customtkinter.CTk()

    # Initialise VideoPlayerUI class. 
    video_player = VideoPlayer(root=root_widget)

    # Start main application event loop.
    root_widget.mainloop()


if __name__ == '__main__':
    main()
    