import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="⏱️ Time Series Based Driving Style Classification", layout="wide")

# ---------------- BACKGROUND ----------------
page_bg = """
<style>
.stApp {
background-image: url("https://images.unsplash.com/photo-1503376780353-7e6692767b70");
background-size: cover;
background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("⏱️ Time Series Based Driving Style Classification")

# ---------------- AUDIO FUNCTION ----------------
def speak(message, loop=False):
    if loop:
        html = f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{message}");
        function repeatSpeech() {{
            speechSynthesis.speak(msg);
        }}
        setInterval(repeatSpeech, 3000);
        </script>
        """
    else:
        html = f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{message}");
        speechSynthesis.speak(msg);
        </script>
        """
    st.components.v1.html(html)

# ---------------- CLASSIFICATION ----------------
def classify(time,speed,acc,brake,steer):
    # Aggressive
    aggressive_conditions = sum([
        speed > 120,
        acc > 3.5,
        steer > 30,
        brake==1 and speed>100
    ])
    if aggressive_conditions >= 2:
        return "Aggressive"

    # Drowsy
    drowsy_conditions = sum([
        speed < 40,
        acc < 1,
        steer < 10,
        brake==0
    ])
    if drowsy_conditions >=3:
        return "Drowsy"

    # Normal
    return "Normal"

# ---------------- DATASET UPLOAD ----------------
st.header("📂 Upload Dataset")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Original Dataset")
    st.dataframe(df)

    required = ["Time","Speed","Acceleration","Brake","Steering"]

    if all(col in df.columns for col in required):
        df["Style"] = df.apply(lambda r: classify(
            r["Time"], r["Speed"], r["Acceleration"], r["Brake"], r["Steering"]
        ), axis=1)

        st.subheader("📊 Predicted Dataset")

        def color_style(val):
            if val=="Normal":
                return "background-color:green;color:white"
            if val=="Drowsy":
                return "background-color:orange;color:white"
            if val=="Aggressive":
                return "background-color:red;color:white"

        st.dataframe(df.style.applymap(color_style, subset=["Style"]))

        # Pie chart
        st.subheader("Pie Chart - Driving Style Distribution")
        pie = px.pie(df, names="Style", color="Style",
                     color_discrete_map={"Normal":"green","Drowsy":"orange","Aggressive":"red"})
        st.plotly_chart(pie,use_container_width=True)

        # Speed vs Time graph
        st.subheader("Speed vs Time Graph")
        line = px.line(df, x="Time", y="Speed", markers=True)
        st.plotly_chart(line,use_container_width=True)
    else:
        st.error("Dataset must contain: Time, Speed, Acceleration, Brake, Steering")

# ---------------- USER INPUT ----------------
st.header("✍ Manual Driving Input")

c1,c2,c3,c4,c5 = st.columns(5)
time_val = c1.number_input("⏱️ Time")
speed = c2.number_input("🚗 Speed (0–360)",0,360)
acc = c3.number_input("⚡ Acceleration (0–13)",0.0,13.0)
brake = c4.selectbox("🛑 Brake",[0,1])
steer = c5.number_input("🛞 Steering (0–65)",0,65)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Predict"):
    style = classify(time_val,speed,acc,brake,steer)

    st.subheader("Prediction Result")
    st.write(f"⏱️ Time: {time_val}")
    st.write(f"🚗 Speed: {speed}")
    st.write(f"⚡ Acceleration: {acc}")
    st.write(f"🛑 Brake: {'Applied' if brake==1 else 'Not Applied'}")
    st.write(f"🛞 Steering: {steer}")

    # Audio alert logic
    if style=="Normal":
        st.success("🟢 Normal Driving")
        speak("Driving is normal", loop=False)
    elif style=="Drowsy":
        st.warning("🟠 Drowsy Driving")
        speak("Warning: Driver is drowsy", loop=True)
    else:
        st.error("🔴 Aggressive Driving")
        speak("Warning: Aggressive driving detected", loop=True)

    st.session_state.history.append({"Time":time_val, "Speed":speed})

# ---------------- SPEED VS TIME GRAPH FOR USER ----------------
if len(st.session_state.history)>0:
    hist_df = pd.DataFrame(st.session_state.history)
    st.subheader("📈 User Input - Speed vs Time")
    fig = px.line(hist_df, x="Time", y="Speed", markers=True)
    st.plotly_chart(fig,use_container_width=True)

# ---------------- SPEEDOMETER ----------------
st.subheader("🚦 Speedometer")
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=speed,
    title={'text':"Speed"},
    gauge={'axis':{'range':[0,360]},'bar':{'color':"blue"}}
))
st.plotly_chart(gauge,use_container_width=True)

# ---------------- VIDEO ----------------
st.header("🎥 Upload Driving Video")
video_file = st.file_uploader("Upload Video (MP4/AVI/MOV)", type=["mp4","avi","mov"])

if video_file:
    st.video(video_file)
    tfile = open("temp.mp4","wb")
    tfile.write(video_file.read())
    cap = cv2.VideoCapture("temp.mp4")

    motion_vals=[]
    frame_nums=[]
    ret, prev = cap.read()
    count=0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        diff = cv2.absdiff(prev, frame)
        motion = np.sum(diff)
        motion_vals.append(motion)
        frame_nums.append(count)
        prev = frame
        count += 1
    cap.release()

    video_df = pd.DataFrame({"Frame":frame_nums,"Motion":motion_vals})
    st.subheader("Video Motion Graph")
    fig = px.line(video_df,x="Frame",y="Motion")
    st.plotly_chart(fig,use_container_width=True)

    motion_score = np.mean(motion_vals)

    if motion_score>5000000:
        st.error("🔴 Aggressive Driving (Video)")
        speak("Warning: Aggressive driving detected", loop=True)
    elif motion_score>2000000:
        st.warning("🟠 Drowsy Driving (Video)")
        speak("Warning: Driver is drowsy", loop=True)
    else:
        st.success("🟢 Normal Driving (Video)")
        speak("Driving is normal", loop=False)