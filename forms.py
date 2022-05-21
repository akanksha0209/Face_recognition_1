from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length
from flask_bootstrap import Bootstrap
from flask_wtf.file import FileField, FileRequired, FileAllowed


class LoginForm(FlaskForm):
    username = StringField(label='Email', validators=[DataRequired()])
    password = PasswordField(label='Password', validators=[DataRequired(), Length(min=1)])
    submit = SubmitField(label='Login')


class StudentForm(FlaskForm):
    first_name = StringField(label='First Name', validators=[DataRequired()])
    last_name = StringField(label='Last Name', validators=[DataRequired()])
    student_id = StringField(label="Student ID", validators=[DataRequired()])
    dob = StringField(label="D.O.B (Format: YYYY-MM-DD)", validators=[DataRequired()])
    semester = StringField(label="Current semester", validators=[DataRequired()])
    branch = StringField(label="Branch", validators=[DataRequired()])
    submit = SubmitField(label='Register')


class ClassAttendance(FlaskForm):
    date = StringField(label='Enter the date for which you want to view class attendance (in YYYY-MM-DD format)',
                       validators=[DataRequired()])
    subject = StringField(label="Enter the subject", validators=[DataRequired()])
    year = StringField(label='Which semester', validators=[DataRequired()])
    branch = StringField(label='Branch', validators=[DataRequired()])
    submit = SubmitField(label='View Class Attendance')


class StudentAttendance(FlaskForm):
    student_id = StringField(label='Enter the Student ID to view attendance',
                             validators=[DataRequired()])
    subject = StringField(label='Enter the subject for which you want to view attendance ',
                          validators=[DataRequired()])

    submit = SubmitField(label='View Student Attendance')


class TeacherForm(FlaskForm):
    first_name = StringField(label='First Name', validators=[DataRequired()])
    last_name = StringField(label='Last Name', validators=[DataRequired()])
    teacher_id = StringField(label="Teachers ID", validators=[DataRequired()])
    # gender = StringField(label='Gender', validators=[DataRequired()])
    class_handled = StringField(label='Subject', validators=[DataRequired()])
    semester = StringField(label='For which semester are you teaching the above subject',
                           validators=[DataRequired()])
    branch = StringField(label="Branch", validators=[DataRequired()])
    submit = SubmitField(label='Register')


class RemoveStudent(FlaskForm):
    student_id = StringField(label='Enter Student ID which you want to remove', validators=[DataRequired()])
    submit = SubmitField(label='Remove')


class RemoveTeacher(FlaskForm):
    teacher_id = StringField(label='Enter Teachers ID which you want to remove', validators=[DataRequired()])
    submit = SubmitField(label='Remove')


class VerifiedImage(FlaskForm):
    submit1 = SubmitField(label='Done')


class RetakeImage(FlaskForm):
    submit2 = SubmitField(label='Retake')

class MyAttendanceForm(FlaskForm):
    student_id=StringField(label='Student ID', validators=[DataRequired()])
    subject=StringField(label='Subject', validators=[DataRequired()])
    submit=SubmitField(label='Submit')