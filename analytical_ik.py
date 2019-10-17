import numpy as np
import math as m
import random

def DHMatrix(t,d,r,a):
    M = np.matrix([
    [m.cos(t) ,-m.sin(t)*m.cos(a) , m.sin(t)*m.sin(a) , r * m.cos(t)  ],
    [m.sin(t) , m.cos(t)*m.cos(a) ,-m.cos(t)*m.sin(a) , r * m.sin(t)  ],
    [0,         m.sin(a),           m.cos(a),           d             ],
    [0,         0,                  0,                  1             ]])
    return M


def ForwardKinematics(q):
    Tbase = DHMatrix(0,0,0,0)
    T1 = DHMatrix(q[0],0.3991,0,-m.pi/2)
    T2 = DHMatrix(q[1]-m.pi/2,0,0.448,0)
    T3 = DHMatrix(q[2],0,0.042,0)
    T4 = DHMatrix(m.pi/2,0,0.451,q[3])
    T5 = DHMatrix(q[4],0,0.082,0)
    T6 = DHMatrix(m.pi/2,0,0,m.pi/2)
    T7 = DHMatrix(q[5],0,0,0)
    bTee = Tbase*T1*T2*T3*T4*T5*T6*T7

    joint_transforms = []
    joint_transforms.append(Tbase)
    joint_transforms.append(Tbase*T1)
    joint_transforms.append(Tbase*T1*T2)
    joint_transforms.append(Tbase*T1*T2*T3*T4)
    joint_transforms.append(Tbase*T1*T2*T3*T4*T5)
    joint_transforms.append(Tbase*T1*T2*T3*T4*T5*T6*T7)

    return joint_transforms, bTee


def InverseKinematics(t):
    l1 = 0.3991
    l2 = 0.448
    l3 = 0.042
    l4 = 0.451
    l5 = 0.082

    theta = m.atan((l5*t[1,2]-t[1,3])/(l5*t[0,2]-t[0,3]))
    if theta>0:
        theta1 = [theta, theta - m.pi]
    else:
        theta1 = [theta, theta + m.pi]


    A = [0, 0]
    A[0] = t[0,3]*m.cos(theta1[0]) + t[1,3]*m.sin(theta1[0]) - l5*t[0,2]*m.cos(theta1[0]) - l5*t[1,2]*m.sin(theta1[0])
    A[1] = t[0,3]*m.cos(theta1[1]) + t[1,3]*m.sin(theta1[1]) - l5*t[0,2]*m.cos(theta1[1]) - l5*t[1,2]*m.sin(theta1[1])
    B = l1 - t[2,3] + l5*t[2,2]


    theta2 = [0,0,0,0]
    for i in range(2):
        a = 2*B*l2
        b = -2*A[i]*l2
        c = l3**2+l4**2-A[i]**2-B**2-l2**2
        theta2[2*i] = 2 * m.atan((b + (a ** 2 + b ** 2 - c ** 2) ** 0.5) / (a + c))
        theta2[2*i+1] = 2 * m.atan((b - (a ** 2 + b ** 2 - c ** 2) ** 0.5) / (a + c))



    theta3 = [0,0,0,0]
    for i in range(2):
        a1 = t[2,3]*m.cos(theta2[2*i]) - l1*m.cos(theta2[2*i]) - l2 - l5*t[2,2]*m.cos(theta2[2*i]) \
            + t[0,3]*m.cos(theta1[i])*m.sin(theta2[2*i]) + t[1,3]*m.sin(theta1[i])*m.sin(theta2[2*i])  \
            - l5*t[1,2]*m.sin(theta1[i])*m.sin(theta2[2*i]) - l5*t[0,2]*m.cos(theta1[i])*m.sin(theta2[2*i])

        b1 = l1*m.sin(theta2[2*i]) - t[2,3]*m.sin(theta2[2*i]) + l5*t[2,2]*m.sin(theta2[2*i]) \
            + t[0,3]*m.cos(theta1[i])*m.cos(theta2[2*i]) + t[1,3]*m.cos(theta2[2*i])*m.sin(theta1[i]) \
            - l5*t[0,2]*m.cos(theta1[i])*m.cos(theta2[2*i]) - l5*t[1,2]*m.cos(theta2[2*i])*m.sin(theta1[i])

        a2 = t[2,3]*m.cos(theta2[2*i+1]) - l1*m.cos(theta2[2*i+1]) - l2 - l5*t[2,2]*m.cos(theta2[2*i+1]) \
            + t[0,3]*m.cos(theta1[i])*m.sin(theta2[2*i+1]) + t[1,3]*m.sin(theta1[i])*m.sin(theta2[2*i+1])  \
            - l5*t[1,2]*m.sin(theta1[i])*m.sin(theta2[2*i+1]) - l5*t[0,2]*m.cos(theta1[i])*m.sin(theta2[2*i+1])

        b2 = l1*m.sin(theta2[2*i+1]) - t[2,3]*m.sin(theta2[2*i+1]) + l5*t[2,2]*m.sin(theta2[2*i+1]) \
            + t[0,3]*m.cos(theta1[i])*m.cos(theta2[2*i+1]) + t[1,3]*m.cos(theta2[2*i+1])*m.sin(theta1[i]) \
            - l5*t[0,2]*m.cos(theta1[i])*m.cos(theta2[2*i+1]) - l5*t[1,2]*m.cos(theta2[2*i+1])*m.sin(theta1[i])

        theta3[2*i]=m.atan2((b1*l3-a1*l4)/(l3**3+l4**2),(a1*l3+b1*l4)/(l3**3+l4**2))
        theta3[2*i+1]=m.atan2((b2*l3-a2*l4)/(l3**3+l4**2),(a2*l3+b2*l4)/(l3**3+l4**2))




    theta5 = [0,0,0,0,0,0,0,0]
    for i in range(2):
        a1 = m.acos(
            t[0, 2] * m.cos(theta1[i]) * m.cos(theta2[2 * i]) * m.cos(theta3[2 * i]) - t[2, 2] * m.cos(theta3[2 * i])
            * m.sin(theta2[2 * i]) - t[2, 2] * m.cos(theta2[2 * i]) * m.sin(theta3[2 * i]) + t[1, 2] * m.cos(theta2[2 * i])
            * m.cos(theta3[2 * i]) * m.sin(theta1[i]) - t[0, 2] * m.cos(theta1[i]) * m.sin(theta2[2 * i])
            * m.sin(theta3[2 * i]) - t[1, 2] * m.sin(theta1[i]) * m.sin(theta2[2 * i]) * m.sin(theta3[2 * i]))

        a2 = m.acos(
            t[0, 2] * m.cos(theta1[i]) * m.cos(theta2[2 * i+1]) * m.cos(theta3[2 * i+1]) - t[2, 2] * m.cos(theta3[2 * i+1])
            * m.sin(theta2[2 * i+1]) - t[2, 2] * m.cos(theta2[2 * i+1]) * m.sin(theta3[2 * i+1]) + t[1, 2] * m.cos(theta2[2 * i+1])
            * m.cos(theta3[2 * i+1]) * m.sin(theta1[i]) - t[0, 2] * m.cos(theta1[i]) * m.sin(theta2[2 * i+1])
            * m.sin(theta3[2 * i+1]) - t[1, 2] * m.sin(theta1[i]) * m.sin(theta2[2 * i+1]) * m.sin(theta3[2 * i+1]))

        theta5[4 * i] = a1
        theta5[4 * i + 1] = -a1
        theta5[4 * i + 2] = a2
        theta5[4 * i + 3] = -a2



    theta4 = [0,0,0,0,0,0,0,0]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if theta5[4*i+2*j+k] != 0 :
                    theta4[4 * i+2*j+k] = m.atan2((t[1, 2] * m.cos(theta1[i]) - t[0, 2] * m.sin(theta1[i])) / m.sin(theta5[4 * i+2*j+k]),
                                                (t[2, 2] * m.cos(theta2[2 * i+j])
                                            * m.cos(theta3[2 * i+j]) - t[2, 2] * m.sin(theta2[2 * i+j]) * m.sin(theta3[2 * i+j]) + t[
                                                0, 2] * m.cos(theta1[i])
                                            * m.cos(theta2[2 * i+j]) * m.sin(theta3[2 * i+j]) + t[0, 2] * m.cos(theta1[i]) * m.cos(
                                                theta3[2 * i+j]) * m.sin(theta2[2 * i+j])
                                            + t[1, 2] * m.cos(theta2[2 * i+j]) * m.sin(theta1[i]) * m.sin(theta3[2 * i+j]) + t[
                                                1, 2] * m.cos(theta3[2 * i+j])
                                            * m.sin(theta1[i]) * m.sin(theta2[2 * i+j])) / (-m.sin(theta5[4 * i+2*j+k])))
                else:
                    theta4[4*i+2*j+k]=random.uniform(-m.pi, m.pi)
                    f=4 * i+2*j+k
                    print('Singularity! Angles of theta 4 & 6 in solution %d are random'%f)



    theta6=[0,0,0,0,0,0,0,0]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                theta6[4*i+2*j+k] = m.atan2(t[1,0]*m.cos(theta1[i])*m.cos(theta4[4*i+2*j+k]) -
                                            t[0,0]*m.cos(theta4[4*i+2*j+k])*m.sin(theta1[i]) + t[2,0]*m.cos(theta2[2 * i+j])
                                            *m.cos(theta3[2 * i+j])*m.sin(theta4[4*i+2*j+k])- t[2,0]*m.sin(theta2[2 * i+j])
                                            *m.sin(theta3[2 * i+j])*m.sin(theta4[4*i+2*j+k]) + t[0,0]*m.cos(theta1[i])
                                            *m.cos(theta2[2 * i+j])*m.sin(theta3[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            + t[0,0]*m.cos(theta1[i])*m.cos(theta3[2 * i+j])*m.sin(theta2[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            + t[1,0]*m.cos(theta2[2 * i+j])*m.sin(theta1[i])*m.sin(theta3[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            + t[1,0]*m.cos(theta3[2 * i+j])*m.sin(theta1[i])*m.sin(theta2[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            ,t[1,1]*m.cos(theta1[i])*m.cos(theta4[4*i+2*j+k]) - t[0,1]*m.cos(theta4[4*i+2*j+k])*m.sin(theta1[i])
                                            + t[2,1]*m.cos(theta2[2 * i+j])*m.cos(theta3[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            - t[2,1]*m.sin(theta2[2 * i+j])*m.sin(theta3[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            + t[0,1]*m.cos(theta1[i])*m.cos(theta2[2 * i+j])*m.sin(theta3[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            + t[0,1]*m.cos(theta1[i])*m.cos(theta3[2 * i+j])*m.sin(theta2[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            + t[1,1]*m.cos(theta2[2 * i+j])*m.sin(theta1[i])*m.sin(theta3[2 * i+j])*m.sin(theta4[4*i+2*j+k])
                                            + t[1,1]*m.cos(theta3[2 * i+j])*m.sin(theta1[i])*m.sin(theta2[2 * i+j])*m.sin(theta4[4*i+2*j+k]))




    solution = np.zeros((8,6))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                solution[4 * i + 2 * j + k, 0] = theta1[i]
                solution[4 * i + 2 * j + k, 1] = theta2[2 * i + j]
                solution[4 * i + 2 * j + k, 2] = theta3[2 * i + j]
                solution[4 * i + 2 * j + k, 3] = theta4[4 * i + 2 * j + k]
                solution[4 * i + 2 * j + k, 4] = theta5[4 * i + 2 * j + k]
                solution[4 * i + 2 * j + k, 5] = theta6[4 * i + 2 * j + k]

    return solution


# numerical IK
def GetJacobian(b_T_ee, joint_transforms):
    J = np.zeros((6,6))
    for i in range(6):
        R = np.zeros((3, 3))
        t = np.zeros((3, 1))
        St = np.zeros((3, 3))
        j_T_ee = np.dot(np.linalg.inv(joint_transforms[i]), b_T_ee)
        ee_T_j = np.linalg.inv(j_T_ee)
        for m in range(3):
            for n in range(3):
                R[m,n] = ee_T_j[m,n]
        for m in range(3):
            t[m] = j_T_ee[m,3]
        St[1][2] = -t[0]
        St[2][1] = t[0]
        St[0][2] = t[1]
        St[2][0] = -t[1]
        St[0][1] = -t[2]
        St[1][0] = t[2]
        Vj = np.hstack((R,-np.dot(R, St)))
        Vj = np.vstack((Vj, np.hstack((np.zeros((3,3)), R))))
        if i == 3:
            J[:, i] = Vj[:, 3]
        else:
            J[:, i] = Vj[:, 5]
    return J

def RotationFromMatrix(matrix):
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    axis = np.real(W[:, i[-1]]).squeeze()
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(axis[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
    elif abs(axis[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
    angle = m.atan2(sina, cosa)
    return angle, axis


def NumericalIK(q0, T):
    q_c = q0
    while True:
        joint_transforms_c, b_T_c = ForwardKinematics(q_c)
        c_T_eecmd = np.dot(np.linalg.inv(b_T_c), T)
        dx = np.array([c_T_eecmd[0,3], c_T_eecmd[1,3], c_T_eecmd[2,3]])
        angle, axis = RotationFromMatrix(c_T_eecmd)
        dori = axis * angle
        dx = np.append(dx, dori)
        if max(abs(dx)) < 10 ** (-8):
            break
        J = GetJacobian(b_T_c, joint_transforms_c)
        Jp = np.linalg.pinv(J)
        dq = np.dot(Jp, dx)
        q_c = q_c + dq
    return q_c


def main():
    
    q = np.random.uniform(-m.pi, m.pi, 6)
    print("q is \n", q)

    joint_transforms, T0 = ForwardKinematics(q)
    print("\nT0 is \n", T0)

    T0[2,3] -= 0.1

    invq1 = InverseKinematics(T0)
    print("\ninvq1 is \n", invq1)

    for qt in invq1:
        Tt = ForwardKinematics(qt)[-1]
        err = np.sum(abs(T0-Tt))
        print("invq1 err is: ", err)

    invq2 = NumericalIK(q, T0)
    print("\ninvq2 is \n", invq2)

    Tt = ForwardKinematics(invq2)[-1]
    err = np.sum(abs(T0-Tt))
    print("\ninvq2 err is: ", err)

if __name__ == "__main__":
    main()
