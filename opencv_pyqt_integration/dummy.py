# creating a photo action to take photo
        click_action = QAction("Click photo", self)
  
        # adding status tip to the photo action
        click_action.setStatusTip("This will capture picture")
  
        # adding tool tip
        click_action.setToolTip("Capture picture")
  
  
        # adding action to it
        # calling take_photo method
        click_action.triggered.connect(self.click_photo)

def click_photo(self):
  
        # time stamp
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
  
        # capture the image and save it on the save path
        self.capture.capture(os.path.join(self.save_path, 
                                          "%s-%04d-%s.jpg" % (
            self.current_camera_name,
            self.save_seq,
            timestamp
        )))
  
        # increment the sequence
        self.save_seq += 1
  