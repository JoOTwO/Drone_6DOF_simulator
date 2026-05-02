import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.animation import FuncAnimation

#1. class 정의
class Drone:
    # 초기값
    def __init__(self):
        # 상태변수 [x, y, z, u, v, w, qw, qx, qy, qz, p, q, r]
        self.state = np.zeros(13)
        self.state[6] = 1.0

        # 물리상수
        self.m = 1.5 # 질량
        self.g = 9.81 # 중력 가속도
        self.l = 0.25 # 중심에서 모터까지의 길이

        # 관성모멘트
        self.Ixx = 0.0134
        self.Iyy = 0.0134
        self.Izz = 0.0264
        self.I = np.array([self.Ixx, self.Iyy, self.Izz])
        self.I_mat = np.diag(self.I) #Inertia Tensor

        # 추진력 및 토크 계수
        self.kt = 2.2e-5 # 추력 계수
        self.kd = 3.5e-7 # 프로펠러 반작용 토크 계수
        kt, kd, l = self.kt, self.kd, self.l
        self.A = np.array([[kt, kt, kt, kt],
                      [-l*kt, -l*kt, l*kt, l*kt],
                      [l*kt, -l*kt, -l*kt, l*kt],
                      [kd, -kd, kd, -kd] #추력, 모멘트 계산 행렬
                        ])
    
    # 모터 추력을 힘으로 바꾸는 함수
    def motor_to_forces(self, motor_Op_sq):
        """
        입력: 모터 1~4에 대한 출력 회전수 제곱 벡터
        출력 : [총추력, 롤모멘트, 피치모멘트, 요모멘트] 벡터
        """
        forces_and_moments = self.A @ motor_Op_sq
        return forces_and_moments
    
    # 동체 좌표계에서의 각을 관성좌표계값으로 변환하는 matrix
    # 쿼터니언으로 변경
    def rotation_matrix_quat(self, quat):
        qw, qx, qy, qz = quat

        R = np.array([[1-2*(qy**2+qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                      [2*(qx*qy + qw*qz), 1-2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                      [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1-2*(qx**2 + qy**2)]
                      ])
        return R
    
    # 상태의 미분값(변화율) 계산 함수
    def dstate(self, state, motor_Op_sq, ext_force):
        """
        입력: 현재 상태(state), 모터 회전수 제곱, 외란
        출력: state의 변화율
        """
        vel_b = state[3:6] # b-frame 속도
        quat = state[6:10] # 쿼터니언
        omega_b = state[10:13] # b-frame 각속도

        # force, moment 계산
        forces = self.motor_to_forces(motor_Op_sq)
        thrust_mag = forces[0]
        torques = forces[1:] #[Tau_x, Tau_y, Tau_z]

        # 회전 행렬
        R = self.rotation_matrix_quat(quat)

        # b-frame에서의 가속도 계산
        Ft_b = np.array([0.0, 0.0, -thrust_mag]) #NED좌표계 부호 고려
        Fg_b = R.T @ np.array([0.0, 0.0, self.m*self.g]) #R_inv = R.T
        Ftol_b = Ft_b + Fg_b + ext_force

        # 병진 운동 방정식 (m(v_dot_b = omega_b x v_b) = F)
        # find v_dot_b
        v_dot_b = Ftol_b / self.m - np.cross(omega_b, vel_b)

        # 회전 운동 방정식 (I*omega_dot = Tau - w x (Iw))
        # find omega_dot
        omega_dot_b = np.linalg.solve(self.I_mat, torques-np.cross(omega_b, self.I_mat @ omega_b))

        # NED-frame(I) 속도 (위치의 시간 변화율)
        vel_I = R @ vel_b
        # 쿼터니언 미분
        p, q, r = omega_b
        qw, qx, qy, qz = quat
        Omega = np.array([[0, -p, -q, -r],
                         [p, 0, r, -q],
                         [q, -r, 0, p],
                         [r, q, -p, 0]])
        q_dot = 0.5 * Omega @ quat
        

        # 전체 미분 벡터
        # [dx, dy, dz, du, dv, dw, dqw, dqx, dqy, dqz, dp, dq, dr]
        d_state = np.hstack((vel_I, v_dot_b, q_dot, omega_dot_b))
        return d_state
        
    
    # 상태 업데이트
    # RK4 사용
    def update(self, motor_Op_sq, dt, ext_force=np.zeros(3)):
        """
        입력: 모터 회전수 제곱(motor_Op_sq), 시간 간격(dt), 외란
        속도, 위치 적분
        """
        # Integration
        # Rk4 사용하여 정밀도 향상
        x = self.state
        k1 = self.dstate(x, motor_Op_sq, ext_force)
        k2 = self.dstate(x + 0.5 * dt * k1, motor_Op_sq, ext_force)
        k3 = self.dstate(x + 0.5 * dt * k2, motor_Op_sq, ext_force)
        k4 = self.dstate(x + dt * k3, motor_Op_sq, ext_force)

        x = x + (dt/6.0) * (k1 +2*k2 +2*k3 +k4)
        
        # 쿼터니언 정규화(Nomalization)
        q_norm = np.linalg.norm(x[6:10])
        x[6:10] /= q_norm

        self.state = x
    
    # 쿼터니언을 오일러 각으로 변환
    # 제어 입력을 위함
    def quat_to_euler(self):
        quat = self.state[6:10]
        qw, qx, qy, qz = quat

        # Roll
        phi = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx**2 + qy**2))
        # Pitch
        sinp = 2*(qw*qy - qz*qx)
        theta = np.arcsin(np.clip(sinp, -1, 1))
        # Yaw
        psi = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy**2 + qz**2))

        return np.array([phi, theta, psi])


# PID
class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0 # 현재까지 쌓인 오차, I Gain용
        self.prev_error = 0.0 # 이전 오차, D Gain용

    #PID term update
    def update(self, error):
        # P항
        P_term = self.Kp * error
        
        # I항
        self.integral += error * self.dt
        limit = 5.0 #anti-windup
        self.integral = np.clip(self.integral, -limit, limit)

        I_term = self.Ki * self.integral

        # D항
        D_term = self.Kd*(error - self.prev_error) / self.dt
        self.prev_error = error

        return P_term + I_term + D_term

    # reset항
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

# Controller
class Controller:
    def __init__(self, dt, drone, ratio):
        self.dt = dt
        self.m = drone.m
        self.g = drone.g
        self.A_inv = np.linalg.pinv(drone.A) #Drone class의 인스턴스
        # 모터 추력 계수 및 프로펠러 반작용 토크 계수
        

        # Outer Loop(위치 제어)용 PID
        self.outer_dt = dt * ratio 
        self.pos_x_pid  = PID(Kp=0.8, Ki=0.0, Kd=1.5, dt=self.outer_dt) 
        self.pos_y_pid  = PID(Kp=0.8, Ki=0.0, Kd=1.5, dt=self.outer_dt)
        self.pos_z_pid  = PID(Kp=0.4, Ki=0.0, Kd=1.5, dt=self.outer_dt) 
        # Inner Loop(자세 제어)용 PID
        self.roll_pid = PID(Kp=10.0, Ki=0.0, Kd=0.6, dt=self.dt)
        self.pitch_pid = PID(Kp=10.0, Ki=0.0, Kd=0.6, dt=self.dt)
        self.yaw_pid = PID(Kp=3.0, Ki=0.0, Kd=0.6, dt=self.dt)
        
   
    def Outer_loop(self, target_pos, current_pos, current_yaw):
        """
        위치 오차를 구해 목표 추력 및 목표 자세를 반환하는 함수
        입력: 목표 위치[x, y, z], 현재 위치[x, y, z], 현재 yaw
        출력: 목표 자세, 목표 추력
        """
        #1. 위치 오차 계산
        err_x = target_pos[0] - current_pos[0]
        err_y = target_pos[1] - current_pos[1]
        err_z = target_pos[2] - current_pos[2]

        #2. PID 계산 -> 목표 가속도(NED Frame)
        acc_n_des = self.pos_x_pid.update(err_x)
        acc_e_des = self.pos_y_pid.update(err_y)
        acc_NED = np.array([acc_n_des, acc_e_des])

        #3. z축 제어 (추력)
        acc_z_des = self.pos_z_pid.update(err_z) 
        target_thrust = self.m * (self.g - acc_z_des)  # 중력 보상 고려, Feedforward
        if target_thrust < 0:
            target_thrust = 0 # 추력은 0보다 작을 수 없음
        
        #4. NED 가속도 -> Heading-aligned Frame에서의 가속도로 변환 
        psi = current_yaw
        Rz_T = np.array([[np.cos(psi), np.sin(psi)],
                       [-np.sin(psi), np.cos(psi)]])
        acc_hd = Rz_T @ acc_NED
        acc_hd_x = acc_hd[0]
        acc_hd_y = acc_hd[1]

        #5. 가속도 -> 각도 변환
        target_theta = -acc_hd_x / self.g 
        target_phi = acc_hd_y / self.g
        
        #각도 제한(Saturation), 선형근사 깨지지 않도록
        limit_angle = np.deg2rad(15)
        target_theta = np.clip(target_theta, -limit_angle, limit_angle)
        target_phi = np.clip(target_phi, -limit_angle, limit_angle)

        return target_phi, target_theta, target_thrust

        
    def Inner_loop(self, target_phi, target_theta, target_psi, current_angles, current_rates): 
        """
        목표 각도를 만들기 위한 필요 토크 출력
        입력: 목표 자세[Roll, Pitch, Yaw], 현재 자세[phi, theta, psi],
        출력: 필요 토크 [tau_x, tau_y, tau_z]
        """
        #1. 현재 상태(각도, 각속도)
        phi = current_angles[0]
        theta = current_angles[1]
        psi = current_angles[2]
        p = current_rates[0]
        q = current_rates[1]
        r = current_rates[2]

        #2. 오차 계산
        err_phi = target_phi - phi
        err_theta = target_theta - theta
        err_psi = target_psi - psi

        # Yaw 각도 보정
        # 예: 목표 179도, 현재 -179도면 실제 차이는 2도인데 수치상 358도가 됨.
        while err_psi > np.pi:
            err_psi -= 2*np.pi
        while err_psi < -np.pi:
            err_psi += 2*np.pi

        #3. PID 계산 -> 목표 토크(Torque)
        torque_x = self.roll_pid.update(err_phi)
        torque_y = self.pitch_pid.update(err_theta)
        torque_z = self.yaw_pid.update(err_psi)

        return torque_x, torque_y, torque_z
    
    # 목표 토크를 모터 회전수로 변경 
    def Compute_MotorOutputs(self, target_thrust, target_torques):
        """
        모터 믹싱 함수
        입력: 목표 추력, 목표 토크 [tau_x, tau_y, tau_z]
        추력: 모터 회전수 제곱
        """
        # 제어 입력
        u_control = np.array([target_thrust,
                              target_torques[0],
                              target_torques[1],
                              target_torques[2]])
        
        motor_Op_sq = self.A_inv @ u_control
        motor_Op_sq = np.maximum(motor_Op_sq, 0) # 출력 클리핑
        
        return motor_Op_sq

#2. MAIN SIMULATION LOOP
if __name__ == "__main__":
    # 설정
    dt = 0.001 # Inner loop 주기
    outer_loop_ratio  = 10 # Outer loop는 10번에 1번만 실행
    sim_time = 25.0
    steps = int(sim_time / dt)

    # 객체 생성
    my_drone = Drone()
    my_controller = Controller(dt, my_drone, ratio = outer_loop_ratio)

    # 외란(Colored Noise) 설정
    current_wind = np.zeros(3) #[x, y, z]방향 힘
    wind_history = []
    alpha = 0.8 # LPF 계수

    # 목표 지점
    waypoints = [[0.0, 0.0, -5.0],
                [5.0, 0.0, -5.0],
                [3.0, 5.0, -5.0],
                [0.0, 4.0, -5.0],
                [0.0, 0.0, -5.0]]
    
    curr_wp_index = 0
    target_pos = waypoints[curr_wp_index]
    acceptance_radius = 0.3 # 도착했다고 인정해줄 거리
    target_yaw = 0.0

    # 데이터 저장용 리스트
    history_pos = []
    history_time = []
    history_ang = []

    print("Simulation Start!")

    #루프 시작
    for i in range(steps):
        # 현재 상태 
        curr_pos = my_drone.state[0:3]
        curr_ang = my_drone.quat_to_euler()
        curr_rate = my_drone.state[10:13] 

        # Noise 생성
        white_noise = np.random.normal(loc=0.0, scale = 4.0, size=3) # (평균, 표준편차, 난수 개수)
        current_wind = alpha * current_wind + (1-alpha) * white_noise
        wind_history.append(current_wind.copy())

        # Waypoints 
        # 도착 확인 후 다음 목적지로 변경
        #1. 현재 위치와 목표 지점 사이 거리 계산
        dist_to_target = np.linalg.norm(target_pos - curr_pos)

        #3. 거리 확인
        if dist_to_target <= acceptance_radius:
            print(f"Waypoint {int(curr_wp_index)+1} 도달! ({i*dt:.2f}초)")

            if curr_wp_index < len(waypoints) -1:
                curr_wp_index += 1
                target_pos = waypoints[curr_wp_index]

            else: 
                pass

        # Outer Loop(위치 -> 목표 자세 & 추력)
        if i % outer_loop_ratio == 0:
            t_phi, t_theta, t_thrust = my_controller.Outer_loop(target_pos, curr_pos, curr_ang[2])

        # Inner Loop(자세 -> 목표 토크)
        t_torques = my_controller.Inner_loop(t_phi, t_theta, target_yaw, curr_ang, curr_rate)

        # Mixer
        motor_sq = my_controller.Compute_MotorOutputs(t_thrust, t_torques)

        # Physics Update
        my_drone.update(motor_sq, dt, ext_force = current_wind)

        # 데이터 저장
        history_pos.append(my_drone.state[0:3].copy())
        history_ang.append(curr_ang.copy())
        history_time.append(i * dt)
    
    print('finish!')


    # 6. 결과 그래프 그리기
    history_pos = np.array(history_pos)
    history_ang = np.array(history_ang)

    # === 결과 데이터 후처리 ===
    # 리스트를 numpy 배열로 변환 (슬라이싱을 위해 필수)
    hist_pos_np = np.array(history_pos)
    hist_ang_np = np.array(history_ang)
    hist_time_np = np.array(history_time)
    wind_np = np.array(wind_history)

    # === 1. 위치 그래프 (Position) ===
    plt.figure(figsize=(12, 10))
    plt.suptitle("Drone Position Response (NED Frame)", fontsize=16)

    # X Position
    plt.subplot(3, 1, 1)
    plt.plot(hist_time_np, hist_pos_np[:, 0], 'r-', label='Current X')
    plt.axhline(y=target_pos[0], color='k', linestyle='--', label='Target X')
    plt.ylabel('X (m)')
    plt.title('X Position')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Y Position
    plt.subplot(3, 1, 2)
    plt.plot(hist_time_np, hist_pos_np[:, 1], 'g-', label='Current Y')
    plt.axhline(y=target_pos[1], color='k', linestyle='--', label='Target Y')
    plt.ylabel('Y (m)')
    plt.title('Y Position')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Z Position (NED 좌표계: 아래가 양수 -> 그래프는 고도(Altitude)로 변환해서 표현)
    plt.subplot(3, 1, 3)
    # 실제 Z값(NED)에 -1을 곱해서 "고도(Altitude)"로 표현
    plt.plot(hist_time_np, -hist_pos_np[:, 2], 'b-', label='Altitude (Current)')
    plt.axhline(y=-target_pos[2], color='k', linestyle='--', label='Target Altitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title('Z Position (Altitude)')
    plt.grid(True)
    plt.legend(loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle 공간 확보
    plt.show()

    # === 2. 자세 그래프 (Attitude in Degrees) ===
    # 오버슈트 분석할 때 자세가 꿀렁이는지 확인해야 합니다.
    plt.figure(figsize=(12, 10))
    plt.suptitle("Drone Attitude Response (Euler Angles)", fontsize=16)

    # Roll
    plt.subplot(3, 1, 1)
    plt.plot(hist_time_np, np.rad2deg(hist_ang_np[:, 0]), 'r-', label='Roll (phi)')
    plt.ylabel('Angle (deg)')
    plt.title('Roll')
    plt.grid(True)
    plt.legend()

    # Pitch
    plt.subplot(3, 1, 2)
    plt.plot(hist_time_np, np.rad2deg(hist_ang_np[:, 1]), 'g-', label='Pitch (theta)')
    plt.ylabel('Angle (deg)')
    plt.title('Pitch')
    plt.grid(True)
    plt.legend()

    # Yaw
    plt.subplot(3, 1, 3)
    plt.plot(hist_time_np, np.rad2deg(hist_ang_np[:, 2]), 'b-', label='Yaw (psi)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.title('Yaw')
    plt.grid(True)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # === 3. 3D 궤적 그래프 (Trajectory) ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 드론 경로 (Z축은 -를 붙여서 위쪽이 하늘이 되게 함)
    ax.plot(hist_pos_np[:, 0], hist_pos_np[:, 1], -hist_pos_np[:, 2], label='Flight Path', linewidth=2)
    
    # 시작점과 목표점 표시
    ax.scatter(0, 0, 0, c='g', marker='o', s=100, label='Start')
    ax.scatter(target_pos[0], target_pos[1], -target_pos[2], c='r', marker='x', s=100, label='Target')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Flight Trajectory')
    ax.legend()
    plt.show()

def get_rotation_matrix(angles):
    """오일러 각(phi, theta, psi) -> 회전 행렬 변환 (Inertial to Body)"""
    phi, theta, psi = angles
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    
    # Inertial -> Body 회전 행렬
    R = np.array([
        [cps*cth,  cps*sth*sph - sps*cph,  cps*sth*cph + sps*sph],
        [sps*cth,  sps*sth*sph + cps*cph,  sps*sth*cph - cps*sph],
        [   -sth,            cth*sph,            cth*cph]
    ])
    return R

def get_drone_arm_points(pos, att, arm_length=0.25):
    '''현재 위치와 자세를 바탕으로 드론 팔의 끝점 좌표 계산'''
    R = get_rotation_matrix(att)
    
    # Body 프레임에서의 팔 끝점 (+형 쿼드콥터 기준)
    p_front = np.array([ arm_length, 0, 0])
    p_back  = np.array([-arm_length, 0, 0])
    p_left  = np.array([0, -arm_length, 0])
    p_right = np.array([0,  arm_length, 0])

    # Body -> Inertial 변환 (R.T 즉, 전치행렬 사용)
    # pos는 (3,) 크기여야 함
    pos = np.array(pos)
    p1_I = pos + R @ p_front
    p2_I = pos + R @ p_back
    p3_I = pos + R @ p_left
    p4_I = pos + R @ p_right
    
    return p1_I, p2_I, p3_I, p4_I

# --- 3D 애니메이션 설정 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Drone 3D Flight Simulation")
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Altitude (m)') 

# 1. 그래프 범위 설정
# waypoints와 history_pos가 정의되어 있다고 가정
wp_np = np.array(waypoints)
all_points = np.vstack([history_pos, wp_np]) 

ax.set_xlim(np.min(all_points[:,0])-1, np.max(all_points[:,0])+1)
ax.set_ylim(np.min(all_points[:,1])-1, np.max(all_points[:,1])+1)
ax.set_zlim(0, np.max(-all_points[:,2]) + 2)

# 2. Waypoint 표시 
ax.plot(wp_np[:, 0], wp_np[:, 1], -wp_np[:, 2], 'rx', markersize=10, markeredgewidth=2, label='Waypoints')

# 각 웨이포인트 옆에 번호(WP0, WP1...) 달아주기
for i, wp in enumerate(waypoints):
    # 텍스트에는 fontweight를 쓸 수 있습니다.
    ax.text(wp[0], wp[1], -wp[2] + 0.3, f"WP{i+1}", color='red', fontsize=9, fontweight='bold')

# 3. 움직이는 요소 초기화
line_arm1, = ax.plot([], [], [], 'b-', linewidth=2, label='Front/Back')
line_arm2, = ax.plot([], [], [], 'r-', linewidth=2, label='Left/Right')
point_center, = ax.plot([], [], [], 'ko', markersize=5)
line_traj, = ax.plot([], [], [], 'g:', linewidth=1, alpha=0.5, label='Trajectory')

ax.legend(loc='upper right')

# 4. 프레임 설정
frame_skip = 50
num_frames = len(history_time) // frame_skip

def update_plot(frame_idx):
    idx = frame_idx * frame_skip
    if idx >= len(history_time): idx = len(history_time) - 1
    
    current_pos = history_pos[idx]
    current_att = history_ang[idx] 
    current_time = history_time[idx]

    # my_drone.l이 없으면 기본값 0.25 사용하도록 처리
    arm_len = getattr(my_drone, 'l', 0.25)
    p1, p2, p3, p4 = get_drone_arm_points(current_pos, current_att, arm_len)

    # 그래프 데이터 업데이트 (Z축 반전: 고도)
    line_arm1.set_data([p1[0], p2[0]], [p1[1], p2[1]])
    line_arm1.set_3d_properties([-p1[2], -p2[2]])
    
    line_arm2.set_data([p3[0], p4[0]], [p3[1], p4[1]])
    line_arm2.set_3d_properties([-p3[2], -p4[2]])

    point_center.set_data([current_pos[0]], [current_pos[1]])
    point_center.set_3d_properties([-current_pos[2]])

    line_traj.set_data(history_pos[:idx+1, 0], history_pos[:idx+1, 1])
    line_traj.set_3d_properties(-history_pos[:idx+1, 2])

    ax.set_title(f"Time: {current_time:.2f}s | Alt: {-current_pos[2]:.1f}m")
    
    return line_arm1, line_arm2, point_center, line_traj

# 5. 실행
ani = FuncAnimation(fig, update_plot, frames=num_frames, 
                    interval=dt*frame_skip*1000, blit=False, repeat=False)

plt.show()