from PyQt5.QtWidgets import QApplication, QWidget

# For accessing command line arguments
# import sys

# Create an instance of QApplication
# If using command line argument use sys.argv 
# If you are not using command line argument use []
# app = QApplication(sys.argv)

app = QApplication([])

# Create a Qt Widget
window = QWidget()

# Widgets without a parent are invisible by default. 
# So, after creating the window object, we must always call .show() 
# to make it visible. You can remove the .show() and run the app, 
# but you'll have no way to quit it!
window.show()

# Start the event loop
app.exec_()