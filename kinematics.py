from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import math as m
import random


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plotRange = 1.5
ax.set_xlim(-0.2,plotRange)
ax.set_ylim(-plotRange*0.5,plotRange*0.5)
ax.set_zlim(-plotRange*0.3,plotRange*0.7)

def dhMatrix(t,d,r,a):
    M = np.matrix([
    [m.cos(t) ,-m.sin(t)*m.cos(a) , m.sin(t)*m.sin(a) , r * m.cos(t)  ],
    [m.sin(t) , m.cos(t)*m.cos(a) ,-m.cos(t)*m.sin(a) , r * m.sin(t)  ],
    [0,         m.sin(a),           m.cos(a),           d             ],
    [0,         0,                  0,                  1             ]])
    return M

def drawFrame(F):
    R = F[0:3,0:3]
    T = F[0:3,3]
    p1 = np.matrix([[0.5],[0],[0]])
    p2 = np.matrix([[0],[0.5],[0]])
    p3 = np.matrix([[0],[0],[0.5]])
    p1 = R*p1
    p2 = R*p2
    p3 = R*p3
    x = T.item(0)
    y = T.item(1)
    z = T.item(2)
    ax.plot([x,x+p1.item(0)],[y,y+p1.item(1)],[z,z+p1.item(2)])
    ax.plot([x,x+p2.item(0)],[y,y+p2.item(1)],[z,z+p2.item(2)])
    ax.plot([x,x+p3.item(0)],[y,y+p3.item(1)],[z,z+p3.item(2)])
    l = 0.1
    ax.plot([x-p3.item(0)*l,x+p3.item(0)*l],
            [y-p3.item(1)*l,y+p3.item(1)*l],
            [z-p3.item(2)*l,z+p3.item(2)*l],
            linewidth=6, color = "k")


def drawSegment(F1,F2):
    T1 = F1[0:3,3]
    T2 = F2[0:3,3]

    ax.plot([T1.item(0),T2.item(0)],
            [T1.item(1),T2.item(1)],
            [T1.item(2),T2.item(2)],
            linewidth=3, color = "k")

def forward_kinematics(q):
    
    Tbase = dhMatrix(0,0,0,0)
    T1 = dhMatrix(q[0],0.3991,0,-m.pi/2)
    T2 = dhMatrix(q[1]-m.pi/2,0,0.448,0)
    T3 = dhMatrix(q[2],0,0.042,0)
    T4 = dhMatrix(m.pi/2,0,0.451,q[3])
    T5 = dhMatrix(q[4],0,0.082,0)
    T6 = dhMatrix(m.pi/2,0,0,m.pi/2)
    T7 = dhMatrix(q[5],0,0,0)
    

    bTee = Tbase*T1*T2*T3*T4*T5*T6*T7

    joint_transforms = []
    joint_transforms.append(Tbase)
    joint_transforms.append(Tbase*T1)
    joint_transforms.append(Tbase*T1*T2)
    joint_transforms.append(Tbase*T1*T2*T3*T4)
    joint_transforms.append(Tbase*T1*T2*T3*T4*T5)
    joint_transforms.append(Tbase*T1*T2*T3*T4*T5*T6*T7)

    return joint_transforms, bTee

def get_jacobian(b_T_ee, joint_transforms):
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

def rotation_from_matrix(matrix):
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

def numerical_IK(b_T_eecmd):
    q_c = np.zeros((6))
    for i in range(6):
        q_c[i] = random.uniform(0, 2 * m.pi)

    while True:
        joint_transforms_c, b_T_c = forward_kinematics(q_c)
        c_T_eecmd = np.dot(np.linalg.inv(b_T_c), b_T_eecmd)
        dx = np.array([c_T_eecmd[0,3], c_T_eecmd[1,3], c_T_eecmd[2,3]])
        angle, axis = rotation_from_matrix(c_T_eecmd)
        dori = axis * angle
        dx = np.append(dx, dori)
        if max(abs(dx)) < 10 ** (-3):
            break
        J = get_jacobian(b_T_c, joint_transforms_c)
        Jp = np.linalg.pinv(J)
        dq = np.dot(Jp, dx)
        q_c = q_c + dq
    return q_c


def main():
    #------------Forward kinematics--------------
    q = np.zeros((6))
    #Set joint angle
    q[0] = 0
    q[1] = 0
    q[2] = 0
    q[3] = 0
    q[4] = 0
    q[5] = 0

    joint_transforms, bTee = forward_kinematics(q)
    print(bTee)
    print("The coordinate of end effector:")
    print("x = ", bTee[0, 3])
    print("y = ", bTee[1, 3])
    print("z = ", bTee[2, 3])

    drawFrame(joint_transforms[0])
    drawFrame(joint_transforms[1])
    drawFrame(joint_transforms[2])
    drawFrame(joint_transforms[3])
    drawFrame(joint_transforms[4])
    drawFrame(joint_transforms[5])
    drawSegment(joint_transforms[0], joint_transforms[1])
    drawSegment(joint_transforms[1], joint_transforms[2])
    drawSegment(joint_transforms[2], joint_transforms[3])
    drawSegment(joint_transforms[3], joint_transforms[4])
    drawSegment(joint_transforms[4], joint_transforms[5])
    plt.show()



    #----------Inverse kinematics--------------

    #Set desired end effector pose
    b_T_eecmd = np.matrix([
            [0,  0, 1, 0.533 ],
            [0,  1, 0, 0     ],
            [-1, 0, 0, 0.9291],
            [0,  0, 0, 1     ]])


    q_c = numerical_IK(b_T_eecmd)
    print("The angle of the joints:")
    print(q_c)

    joint_transforms, bTee = forward_kinematics(q_c)
    drawFrame(joint_transforms[0])
    drawFrame(joint_transforms[1])
    drawFrame(joint_transforms[2])
    drawFrame(joint_transforms[3])
    drawFrame(joint_transforms[4])
    drawFrame(joint_transforms[5])
    drawSegment(joint_transforms[0], joint_transforms[1])
    drawSegment(joint_transforms[1], joint_transforms[2])
    drawSegment(joint_transforms[2], joint_transforms[3])
    drawSegment(joint_transforms[3], joint_transforms[4])
    drawSegment(joint_transforms[4], joint_transforms[5])
    plt.show()



if __name__ == "__main__":
    main()
