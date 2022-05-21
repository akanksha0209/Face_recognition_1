import time
from flask import Flask, render_template, url_for, request, flash, redirect, Response
from datetime import datetime
import cv2
import os
import numpy as np
import face_recognition
from flask_bootstrap import Bootstrap
from forms import LoginForm, StudentForm, ClassAttendance, StudentAttendance, TeacherForm, RemoveStudent, RemoveTeacher, \
    VerifiedImage, RetakeImage, MyAttendanceForm
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String

# declare the app
app = Flask(__name__)

# database configurations
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///students.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# secret key
app.secret_key = "akanksha"
Bootstrap(app)

# opencv's face detector and pretrained model
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
# gender detection and pretrained model
gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"
# declare opencv's dnn to work on the models
face_net = cv2.dnn.readNet(face_model, face_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

genderList = ['Male', 'Female']

face_classifier = cv2.CascadeClassifier("haarcascade_face.xml")


# student database
class Student(db.Model):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    student_id = Column(String, nullable=False)
    dob = Column(String, nullable=False)
    gender = Column(String, nullable=True)
    semester = Column(String, nullable=True)
    branch = Column(String, nullable=True)
    photo_url = Column(String, nullable=True)

    def __repr__(self):
        return f'{self.student_id}'

    def add_student(self, first_name, last_name,
                    student_id, dob, gender, semester, branch, photo_url):
        new_user = Student(first_name=first_name, last_name=last_name,
                           student_id=student_id, gender=gender, dob=dob,
                           semester=semester, branch=branch, photo_url=photo_url)

        db.session.add(new_user)
        db.session.commit()

    def remove_student(self, student_id):
        delete_user = Student.query.filter_by(student_id=student_id).first()
        db.session.delete(delete_user)
        db.session.commit()


db.create_all()

today = datetime.now().date()


# attendance table in database
class Attendance(db.Model):
    id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    student_id = Column(String, nullable=False)
    subject = Column(String, nullable=True)
    semester = Column(String, nullable=True)
    branch = Column(String, nullable=True)
    date = Column(String, nullable=True)

    def __repr__(self):
        return f'{self.student_id}'

    def add_attendance_student(self, first_name, last_name, student_id, subject, semester, branch, date):
        new_student = Attendance(first_name=first_name, last_name=last_name, student_id=student_id, subject=subject,
                                 semester=semester, branch=branch, date=date)

        db.session.add(new_student)
        db.session.commit()


db.create_all()


# teacher model in database
class Teacher(db.Model):
    __tablename__ = 'teachers'

    id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    teacher_id = Column(String, nullable=False)
    gender = Column(String, nullable=True)
    class_handled = Column(String, nullable=True)
    semester = Column(String, nullable=True)
    branch = Column(String, nullable=True)
    photo_url = Column(String, nullable=True)
    no_of_classes = Column(Integer, nullable=True)

    def __repr__(self):
        return f'{self.teacher_id}'

    def add_teacher(self, first_name, last_name,
                    teacher_id, gender, class_handled, semester, branch, photo_url):
        new_user = Teacher(first_name=first_name, last_name=last_name,
                           teacher_id=teacher_id, gender=gender, class_handled=class_handled,
                           semester=semester, branch=branch, photo_url=photo_url)

        db.session.add(new_user)
        db.session.commit()

    def remove_teacher(self, teacher_id):
        delete_user = Teacher.query.filter_by(teacher_id=teacher_id).first()
        db.session.delete(delete_user)
        db.session.commit()


db.create_all()


# timetable in database
class TimeTable(db.Model):
    __tablename__ = 'timetable'
    id = Column(Integer, primary_key=True)
    branch = Column(String, nullable=True)
    semester = Column(String, nullable=True)
    day = Column(String, nullable=True)
    period = Column(String, nullable=True)
    subject = Column(String, nullable=True)
    teacher = Column(String, nullable=True)

    def __repr__(self):
        return f'{self.subject}'

    def create_timetable(self):
        create = TimeTable.query.all()
        if len(create) == 0:
            c1 = TimeTable(branch='CSE', semester='3rd', day='0', period='1st-period',
                           subject='DSA', teacher='Teacher1')
            c2 = TimeTable(branch='CSE', semester='3rd', day='0', period='2nd-period',
                           subject='OS', teacher='Teacher2')
            c3 = TimeTable(branch='CSE', semester='3rd', day='0', period='3rd-period',
                           subject='Math II', teacher='Teacher3')
            c4 = TimeTable(branch='CSE', semester='3rd', day='0', period='4th-period',
                           subject='DMS', teacher='Teacher4')
            c5 = TimeTable(branch='CSE', semester='3rd', day='1', period='1st-period',
                           subject='DMS', teacher='Teacher4')
            c6 = TimeTable(branch='CSE', semester='3rd', day='1', period='2nd-period',
                           subject='Math II', teacher='Teacher3')
            c7 = TimeTable(branch='CSE', semester='3rd', day='1', period='3rd-period',
                           subject='OS', teacher='Teacher2')
            c8 = TimeTable(branch='CSE', semester='3rd', day='1', period='4th-period',
                           subject='DSA-Lab', teacher='Teacher1')
            c9 = TimeTable(branch='CSE', semester='3rd', day='2', period='1st-period',
                           subject='FCSD', teacher='Teacher5')
            c10 = TimeTable(branch='CSE', semester='3rd', day='2', period='2nd-period',
                            subject='DMS', teacher='Teacher4')
            c11 = TimeTable(branch='CSE', semester='3rd', day='2', period='3rd-period',
                            subject='DSA', teacher='Teacher1')
            c12 = TimeTable(branch='CSE', semester='3rd', day='2', period='4th-period',
                            subject='OS-Lab', teacher='Teacher2')
            c13 = TimeTable(branch='CSE', semester='3rd', day='3', period='1st-period',
                            subject='OS', teacher='Teacher2')
            c14 = TimeTable(branch='CSE', semester='3rd', day='3', period='2nd-period',
                            subject='FCSD-Lab', teacher='Teacher5')
            c15 = TimeTable(branch='CSE', semester='3rd', day='3', period='3rd-period',
                            subject='DMS', teacher='Teacher4')
            c16 = TimeTable(branch='CSE', semester='3rd', day='3', period='4th-period',
                            subject='DSA', teacher='Teacher1')
            c17 = TimeTable(branch='CSE', semester='3rd', day='4', period='1st-period',
                            subject='Math II', teacher='Teacher3')
            c18 = TimeTable(branch='CSE', semester='3rd', day='4', period='2nd-period',
                            subject='FCSD', teacher='Teacher5')
            c19 = TimeTable(branch='CSE', semester='3rd', day='4', period='3rd-period',
                            subject='DMS', teacher='Teacher4')
            c20 = TimeTable(branch='CSE', semester='3rd', day='5', period='4th-period',
                            subject='OS', teacher='Teacher2')

            e1 = TimeTable(branch='ECE', semester='3rd', day='0', period='1st-period',
                           subject='ADDC', teacher='Teacher6')
            e2 = TimeTable(branch='ECE', semester='3rd', day='0', period='2nd-period',
                           subject='EF', teacher='Teacher7')
            e3 = TimeTable(branch='ECE', semester='3rd', day='0', period='3rd-period',
                           subject='NA', teacher='Teacher8')
            e4 = TimeTable(branch='ECE', semester='3rd', day='0', period='4th-period',
                           subject='AMC', teacher='Teacher9')
            e5 = TimeTable(branch='ECE', semester='3rd', day='1', period='1st-period',
                           subject='DIT', teacher='Teacher10')
            e6 = TimeTable(branch='ECE', semester='3rd', day='1', period='2nd-period',
                           subject='DIT', teacher='Teacher10')
            e7 = TimeTable(branch='ECE', semester='3rd', day='1', period='3rd-period',
                           subject='AMC-Lab', teacher='Teacher9')
            e8 = TimeTable(branch='ECE', semester='3rd', day='1', period='4th-period',
                           subject='NA', teacher='Teacher8')
            e9 = TimeTable(branch='ECE', semester='3rd', day='2', period='1st-period',
                           subject='EF', teacher='Teacher7')
            e10 = TimeTable(branch='ECE', semester='3rd', day='2', period='2nd-period',
                            subject='ADDC', teacher='Teacher6')
            e11 = TimeTable(branch='ECE', semester='3rd', day='2', period='3rd-period',
                            subject='NA-Lab', teacher='Teacher8')
            e12 = TimeTable(branch='ECE', semester='3rd', day='2', period='4th-period',
                            subject='DIT', teacher='Teacher10')
            e13 = TimeTable(branch='ECE', semester='3rd', day='3', period='1st-period',
                            subject='ADDC-Lab', teacher='Teacher6')
            e14 = TimeTable(branch='ECE', semester='3rd', day='3', period='2nd-period',
                            subject='DIT', teacher='Teacher10')
            e15 = TimeTable(branch='ECE', semester='3rd', day='3', period='3rd-period',
                            subject='NA', teacher='Teacher8')
            e16 = TimeTable(branch='ECE', semester='3rd', day='3', period='4th-period',
                            subject='EF-Lab', teacher='Teacher7')
            e17 = TimeTable(branch='ECE', semester='3rd', day='4', period='1st-period',
                            subject='DIT', teacher='Teacher10')
            e18 = TimeTable(branch='ECE', semester='3rd', day='4', period='2nd-period',
                            subject='ADDC', teacher='Teacher6')
            e19 = TimeTable(branch='ECE', semester='3rd', day='4', period='3rd-period',
                            subject='AMC', teacher='Teacher9')
            e20 = TimeTable(branch='ECE', semester='3rd', day='5', period='4th-period',
                            subject='EF', teacher='Teacher7')

            db.session.add(c1)
            db.session.add(c2)
            db.session.add(c3)
            db.session.add(c4)
            db.session.add(c5)
            db.session.add(c6)
            db.session.add(c7)
            db.session.add(c8)
            db.session.add(c9)
            db.session.add(c10)
            db.session.add(c11)
            db.session.add(c12)
            db.session.add(c13)
            db.session.add(c14)
            db.session.add(c15)
            db.session.add(c16)
            db.session.add(c17)
            db.session.add(c18)
            db.session.add(c19)
            db.session.add(c20)
            db.session.add(e1)
            db.session.add(e2)
            db.session.add(e3)
            db.session.add(e4)
            db.session.add(e5)
            db.session.add(e6)

            db.session.add(e7)
            db.session.add(e8)
            db.session.add(e9)
            db.session.add(e10)
            db.session.add(e11)
            db.session.add(e12)
            db.session.add(e13)
            db.session.add(e14)
            db.session.add(e15)
            db.session.add(e16)
            db.session.add(e17)
            db.session.add(e18)
            db.session.add(e19)
            db.session.add(e20)

            db.session.commit()


db.create_all()

c = TimeTable()
c.create_timetable()
names = []


class StudentLogins(db.Model):
    __tablename__ = 'student login details'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

    def __repr__(self):
        return f'{self.student_id}'

    def add_student_login_detail(self, student_id, password):
        new_user = StudentLogins(student_id=student_id, password=password)

        db.session.add(new_user)
        db.session.commit()

    def remove(self, student_id):
        remove_user = StudentLogins(student_id=student_id).first()
        db.session.delete(remove_user)
        db.session.commit()


db.create_all()


# function to open the webcam in the web application
def gen(camera):
    encodeListKnown = camera.get_encodings()

    while True:
        image = camera.get_frame(encodeListKnown)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


# function to mark attendance of students for a particular subject
def mark_attendance():
    subject = ''
    period = ''
    teacher_name = ''
    teachers = Teacher.query.all()

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    if '09:00:00' <= current_time < '10:30:00':
        period = '1st-period'
    if '10:30:00' <= current_time < '12:00:00':
        period = '2nd-period'
    if '12:00:00' <= current_time < '13:30:00':
        period = '3rd-period'
    if '14:30:00' <= current_time < '16:00:00':
        period = '4th-period'
    current_day = datetime.today().weekday()
    # 0 is monday and 6 is sunday

    attendance = Attendance()
    for name in names:  # note: name contains the unique id of a teacher/student
        for teacher in teachers:
            if teacher.teacher_id == name:
                subject = TimeTable.query.filter_by(teacher=name, period=period,
                                                    day=current_day).first()  # searches for
                # which subject is being taught in the particular class
                teacher_name = name
                print(f'subject is {subject}')

    # updates the total no of classes completed in the subject
    no_of_classes_update = Teacher.query.filter_by(teacher_id=teacher_name,
                                                   semester=subject.semester,
                                                   class_handled=f'{subject}').first()

    if no_of_classes_update.no_of_classes is None:
        no_of_classes_update.no_of_classes = 1
    else:
        no_of_classes_update.no_of_classes = no_of_classes_update.no_of_classes + 1

    db.session.commit()
    # marks attendance
    for name in names:
        exists = Attendance.query.filter_by(subject=f'{subject}', date=today, student_id=name).first()

        f = exists

        if f is None:

            student_name = Student.query.filter_by(student_id=f'{name}').first()
            if student_name is None:  # if its a teacherid doesn't mark attendance
                pass
            else:
                attendance.add_attendance_student(student_name.first_name, student_name.last_name,
                                                  student_name.student_id,
                                                  f'{subject}', student_name.semester, student_name.branch,
                                                  today)


# camera class to recognise faces
class Camera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        path = r'static/images/Students_and_Teachers/'
        self.images = []
        self.classNames = []

        for cl in os.listdir(path):
            curImg = cv2.imread(f'{path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])

    def __del__(self):

        self.video.release()

    def findEncodings(self):  # gets face encodings
        encodeListKnown = []

        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeListKnown.append(encode)
            print("complete")

        return encodeListKnown

    def get_encodings(self):
        encodeListKnown = self.findEncodings()
        return encodeListKnown

    # gets the frame from webcam and recognises the person if the person is pre registered
    def get_frame(self, encodeListKnown):
        ret, image = self.video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_classifier.detectMultiScale(gray, 1.3, 5)
        imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesinCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesinCurFrame)
        for encode_face, face_loc in zip(encodesCurFrame, facesinCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encode_face)
            facedis = face_recognition.face_distance(encodeListKnown, encode_face)

            matchIndex = np.argmin(facedis)
            if matches[matchIndex]:
                name = self.classNames[matchIndex]
                name = name.split('.')[1]

                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                names.append(name)

        ret, jpg = cv2.imencode('.jpg', image)
        return jpg.tobytes()


# gets the coordinates of the rectangle inside which the person's face is located
def get_face_box(face_net, frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True)
    face_net.setInput(blob)
    detection = face_net.forward()
    boxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frame_width)
            y1 = int(detection[0, 0, i, 4] * frame_height)
            x2 = int(detection[0, 0, i, 5] * frame_width)
            y2 = int(detection[0, 0, i, 6] * frame_height)
            boxes.append([x1, y1, x2, y2])

    return frame, boxes


# detects the faces using a pretrained model
def detect_faces(id_user, first_name, last_name):
    cap = cv2.VideoCapture(0)
    padding = 20  # padding to make sure the person's face is clearly visible
    gender = ''
    file_name_path = ''

    while True:

        ret, frame = cap.read()

        frame, boxes = get_face_box(face_net, frame)
        for box in boxes:
            face = frame[max(0, box[1] - padding):min(box[3] + padding, frame.shape[0] - 1),
                   max(0, box[0] - padding):min(box[2] + padding, frame.shape[1] - 1)]
            # adds the image of the student to the folder mentioned in file_name_path
            file_name_path = f"static/images/Students_and_Teachers/{first_name} {last_name}.{id_user}.jpg"
            cv2.imwrite(file_name_path, face)
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_pred = gender_net.forward()
            gender = genderList[gender_pred[0].argmax()]
        if cv2.waitKey(5000):
            break

    cap.release()
    cv2.destroyAllWindows()
    return file_name_path, gender


# reinitialse the list names after each class attendance is taken and attendance for that class is marked
def reinitialise(names):
    names.clear()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/main_page')
def main_page():
    return render_template('home.html')


@app.route('/student_home_page')
def student_home_page():
    return render_template('student_home.html')


# login page
@app.route('/login', methods=["GET", "POST"])
def login():
    login_form = LoginForm()
    students = StudentLogins.query.all()
    print(students)

    if login_form.validate_on_submit():
        if login_form.username.data == 'a' and login_form.password.data == '1':

            return render_template('home.html')
        else:
            for student in students:
                print(student)
                if login_form.username.data == f'{student}' and login_form.password.data == student.password:

                    return render_template('student_home.html')
                else:
                    return render_template('login.html', form=login_form)

    return render_template('login.html', form=login_form)


@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')


# used to open the web came on the web application
@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/return_to_home')
def return_home():
    mark_attendance()
    reinitialise(names)
    return render_template('home.html')


# used by student to view attendance
@app.route('/view_my_attendance', methods=["GET", "POST"])
def view_my_attendance():
    student_attendance_form = StudentAttendance()
    if student_attendance_form.validate_on_submit():
        student_id_required = student_attendance_form.student_id.data
        subject = student_attendance_form.subject.data
        details = Student.query.filter_by(student_id=student_id_required).first()
        year = details.semester
        branch = details.branch

        classes_attended = Attendance.query.filter_by(student_id=student_id_required, subject=subject)
        classes = Teacher.query.filter_by(class_handled=subject, semester=year,
                                          branch=branch).first()
        total_no_of_classes = classes.no_of_classes

        no_of_classes = Attendance.query.filter_by(student_id=student_id_required, subject=subject).all()
        name = Attendance.query.filter_by(student_id=student_id_required).first()
        no_of_classes_attended = len(no_of_classes)
        return render_template('student_attendance.html', all_classes_attended=classes_attended,
                               name=name, no_of_classes_attended=no_of_classes_attended,
                               total_no_of_classes=total_no_of_classes)

    return render_template('view_attendance.html',
                           student_attendance_form=student_attendance_form)


# used by teacher to view attendance
@app.route('/view_attendance', methods=["GET", "POST"])
def view_attendance():
    class_attendance_form = ClassAttendance()
    student_attendance_form = StudentAttendance()
    if class_attendance_form.validate_on_submit():  # if teacher wants the class attendance for a particular day
        date_required = class_attendance_form.date.data
        subject = class_attendance_form.subject.data
        branch = class_attendance_form.branch.data
        year = class_attendance_form.year.data

        students_present_on_date = Attendance.query.filter_by(date=date_required, subject=subject,
                                                              semester=year, branch=branch)

        date = Attendance.query.filter_by(date=date_required).first()

        return render_template('class_attendance.html', all_students_present=students_present_on_date, date=date)
    if student_attendance_form.validate_on_submit():  # if teacher wants the student attendance for a particular subject
        student_id_required = student_attendance_form.student_id.data
        subject = student_attendance_form.subject.data
        details = Student.query.filter_by(student_id=student_id_required).first()
        year = details.semester
        branch = details.branch

        classes_attended = Attendance.query.filter_by(student_id=student_id_required, subject=subject)
        classes = Teacher.query.filter_by(class_handled=subject, semester=year,
                                          branch=branch).first()
        total_no_of_classes = classes.no_of_classes

        no_of_classes = Attendance.query.filter_by(student_id=student_id_required, subject=subject).all()
        name = Attendance.query.filter_by(student_id=student_id_required).first()
        no_of_classes_attended = len(no_of_classes)

        return render_template('student_attendance.html', all_classes_attended=classes_attended,
                               name=name, no_of_classes_attended=no_of_classes_attended,
                               total_no_of_classes=total_no_of_classes)

    return render_template('view_attendance.html', class_attendance_form=class_attendance_form,
                           student_attendance_form=student_attendance_form)


# when faculty member is registering the student during admission process
@app.route('/add_new_student', methods=["GET", "POST"])
def add_new_student():
    form = StudentForm()
    remove_form = RemoveStudent()
    if form.validate_on_submit():
        first_name = form.first_name.data
        print(first_name)
        last_name = form.last_name.data
        student_id = form.student_id.data
        dob = form.dob.data
        semester = form.semester.data
        branch = form.branch.data
        photo_url, gender = detect_faces(student_id, first_name, last_name)

        s = Student()
        # adds new student
        s.add_student(first_name=first_name, last_name=last_name,
                      student_id=student_id, dob=dob, gender=gender, semester=semester,
                      branch=branch, photo_url=photo_url)
        l = StudentLogins()
        password = f'{first_name}{dob}{last_name}'
        l.add_student_login_detail(student_id=student_id, password=password)
        time.sleep(2)

        return redirect(url_for('verify_image', person_id=student_id))
    if remove_form.validate_on_submit():
        student_id = form.student_id.data
        s = Student()
        student_image = s.query.filter_by(student_id=student_id).first()
        # removes the image of the student
        os.remove(
            f'static/images/Students_and_Teachers/{student_image.first_name} {student_image.last_name}.{student_id}.jpg')
        # removes student details from database
        s.remove_student(student_id=student_id)
        l = StudentLogins()
        l.remove(student_id=student_id)

        return render_template('home.html')

    return render_template('add_new_student.html', form=form, remove_form=remove_form)


# when faculty member is verifying the image of the student or teacher
@app.route('/verify_image/<person_id>', methods=["GET", "POST"])
def verify_image(person_id):
    verified_image_form = VerifiedImage()
    retake_image_form = RetakeImage()
    s = Student()
    image = s.query.filter_by(student_id=person_id).first()
    if image is None:
        image = Teacher.query.filter_by(teacher_id=person_id).first()

    # if user wants to retake image
    if retake_image_form.submit2.data and retake_image_form.validate():
        # removes the previously saved image and captures a new image

        os.remove(
            f'static/images/Students_and_Teachers/{image.first_name} {image.last_name}.{person_id}.jpg')
        time.sleep(2)
        photo_url, gender = detect_faces(person_id, image.first_name, image.last_name)

        if image is not None:  # if student
            image.gender = gender

        else:  # if teacher
            t = Teacher()
            teacher_image = t.query.filter_by(teacher_id=person_id).first()
            teacher_image.gender = gender
        db.session.commit()
        return redirect(url_for('verify_image', person_id=person_id))
    # user is happy with the image captured
    if verified_image_form.submit1.data and verified_image_form.validate():
        return redirect(url_for('main_page'))

    return render_template('verify_image.html', person_id=person_id, first_name=image.first_name,
                           last_name=image.last_name, verified_image_form=verified_image_form,
                           retake_image_form=retake_image_form)


@app.route('/add_new_teacher', methods=["GET", "POST"])
def add_new_teacher():
    form = TeacherForm()
    remove_form = RemoveTeacher()
    if form.validate_on_submit():
        first_name = form.first_name.data
        print(first_name)
        last_name = form.last_name.data
        teacher_id = form.teacher_id.data
        semester = form.semester.data
        branch = form.branch.data
        class_handled = form.class_handled.data
        photo_url, gender = detect_faces(teacher_id, first_name, last_name)

        t = Teacher()
        # adds new teacher
        t.add_teacher(first_name=first_name, last_name=last_name,
                      teacher_id=teacher_id, gender=gender, class_handled=class_handled,
                      semester=semester, branch=branch, photo_url=photo_url)

        return redirect(url_for('verify_image', person_id=teacher_id))
    if remove_form.validate_on_submit():
        teacher_id = form.teacher_id.data
        t = Teacher()
        teacher_image = t.query.filter_by(teacher_id=teacher_id).first()
        os.remove(
            f'static/images/Students_and_Teachers/{teacher_image.first_name} {teacher_image.last_name}.{teacher_image}.jpg')

        t.remove_teacher(teacher_id=teacher_id)
        return render_template('home.html')
    return render_template('add_new_teacher.html', form=form, remove_form=remove_form)


@app.route('/view_student_details')
def view_student_details():
    s = Student()
    students = s.query.all()
    return render_template('view_student_details.html', students=students)


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
