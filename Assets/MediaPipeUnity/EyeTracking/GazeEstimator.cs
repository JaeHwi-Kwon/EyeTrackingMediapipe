using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Mediapipe;
using Mediapipe.Unity;
using Mediapipe.Unity.CoordinateSystem;
using UnityEngine.UI;
using System.Linq;
using OpenCvSharp;
using UnityEngine.UIElements;
using Unity.VisualScripting;
using System;
using UnityEditor.AssetImporters;
using Mediapipe.Unity.EyeTrackingSystem;

public class GazeEstimator : MonoBehaviour
{
    [SerializeField] private RawImage screen;
    [SerializeField] private GameObject cube;
    [SerializeField] private GameObject canvas;
    [SerializeField] private float width;
    [SerializeField] private float height;
    [SerializeField] private GameObject cursor;
    [SerializeField] private int BufSizeForSmoothing;

    private Vector3 vec;
    
    private Vector3 TemporarycalibratedDirection;
    private float XWeight1, XWeight2, YWeight1, YWeight2, XBias, YBias;
    private bool isCalibreated = false;

    private UnityEngine.Rect screenRect;
    private GameObject[] spheresForLandmarks = new GameObject[468];

    private int[] LeftEyeIndices = new int[]{362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398};
    private int[] RightEyeIndices = new int[] { 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 };

    private int[] LeftIrisIndices = new int[] { 474, 475, 476, 477 };
    private int[] RightIrisIndices = new int[] { 469, 470, 471, 472 };

    private Smoothing smoothing;

    // Start is called before the first frame update
    void Start()
    {
        //screenRect = screen.GetComponent<RectTransform>().rect;
        //for (int i = 0; i < 468; i++)
        //{
        //    spheresForLandmarks[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        //    spheresForLandmarks[i].transform.SetParent(canvas.transform);
        //    spheresForLandmarks[i].transform.localScale = new Vector3(5, 5, 5);
        //}

        smoothing = new Smoothing(BufSizeForSmoothing);
    }

    // Update is called once per frame
    void Update()
    {

        if (isCalibreated)
        {
            MoveCursorByGaze(XWeight1, XWeight2, XBias, YWeight1, YWeight2, YBias);
        }
    }

    public Vector2 EyeTracker(NormalizedLandmarkList landmarks)
    {
        // 눈 landmark 가져오기
        Vector2[] leyepoints = new Vector2[LeftEyeIndices.Length];
        for(int i = 0; i < LeftEyeIndices.Length; i++)
        {
            leyepoints[i] = new Vector2(landmarks.Landmark[LeftEyeIndices[i]].X*width, 
                                        height-landmarks.Landmark[LeftEyeIndices[i]].Y*height);
        }
        Vector2[] reyepoints = new Vector2[RightEyeIndices.Length];
        for (int i = 0; i < RightEyeIndices.Length; i++)
        {
            reyepoints[i] = new Vector2(landmarks.Landmark[RightEyeIndices[i]].X * width,
                                        height - landmarks.Landmark[RightEyeIndices[i]].Y * height);
        }
        Vector2[] leyeRect = new Vector2[2];
        Vector2[] reyeRect = new Vector2[2];
        
        // 눈 영역 구하기
        leyeRect = GetEyeRectPoint(leyepoints);
        reyeRect = GetEyeRectPoint(reyepoints);

        // 눈 영역의 중점 구하기
        Vector2 leyeRectCenter = GetCenterPoint(leyeRect);
        Vector2 reyeRectCenter = GetCenterPoint(reyeRect);

        // 눈 영역 가로세로 사이즈 구하기
        Vector2 leyeRectSize = new Vector2(leyeRect[1].x - leyeRect[0].x, leyeRect[0].y - leyeRect[1].y);
        Vector2 reyeRectSize = new Vector2(reyeRect[1].x - reyeRect[0].x, reyeRect[0].y - reyeRect[1].y);


        // 눈동자 landmark 가져오기
        Vector2[] LIrisPoints = new Vector2[4];
        Vector2[] RIrisPoints = new Vector2[4];
        for (int i = 0;i<4; i++)
        {
            LIrisPoints[i] = new Vector2(landmarks.Landmark[LeftIrisIndices[i]].X*width,
                                         height - landmarks.Landmark[LeftIrisIndices[i]].Y*height);
        }
        for (int i = 0; i < 4; i++)
        {
            RIrisPoints[i] = new Vector2(landmarks.Landmark[RightIrisIndices[i]].X * width,
                                         height - landmarks.Landmark[RightIrisIndices[i]].Y * height);
        }

        // 각 눈동자의 중점
        Vector2 LIrisCenter = GetCenterPoint(LIrisPoints);
        Vector2 RIrisCenter = GetCenterPoint(RIrisPoints);

        // 눈동자 위치와 눈 중점간 차이 비교로 눈 이동 확인
        Vector2 LeyePos = LIrisCenter - leyeRectCenter;
        Vector2 ReyePos = RIrisCenter - reyeRectCenter;

        // 눈 이동거리 노멀라이즈
        LeyePos = new Vector2(LeyePos.x / (leyeRectSize.x/2), LeyePos.y / (leyeRectSize.y/2));
        ReyePos = new Vector2(ReyePos.x / (reyeRectSize.x/2), ReyePos.y / (reyeRectSize.y/2));


        //Debug.Log(LeyePos + "\n" + ReyePos);

        return LeyePos;
    }

    private Vector2 GetCenterPoint(Vector2[] points)
    {
        Vector2 center = new Vector2(0,0);

        foreach (Vector2 point in points)
        {
            center = new Vector2(center.x + point.x, center.y + point.y);
        }
        center = center/points.Length;

        return center;
    }

    private Vector2[] GetEyeRectPoint(Vector2[] points)
    {

        //[(left,top), (right,bottom)]
        Vector2[] rect = new Vector2[] { points[0], points[0] };

        for(int i=0;i< points.Length; i++)
        {
            if (rect[0].x > points[i].x)
                rect[0].x = points[i].x;
            if (rect[1].x < points[i].x)
                rect[1].x = points[i].x;

            if (rect[0].y < points[i].y)
                rect[0].y = points[i].y;
            if (rect[1].y > points[i].y)
                rect[1].y = points[i].y;
        }

        return rect;
    }

    public Vector2 HeadPoseTracker(NormalizedLandmarkList landmarks)
    {
        if(landmarks == null) { Debug.Log("Empty!"); }

        Point3f[] objectPoints = new Point3f[6];


        // Head Pose 추적 대표 점 Landmark (3D 월드 좌표계)
        //NoseTop
        objectPoints[0] = new Point3f(0, 0, 0);
        //LeftEye
        objectPoints[1] = new Point3f(-225, 170, -135);
        //RightEye
        objectPoints[2] = new Point3f(255, 170, -135);
        //MouthLeft
        objectPoints[3] = new Point3f(-150, -150, -125);
        //MouthRight
        objectPoints[4] = new Point3f(150, -150, -125);
        //Chin
        objectPoints[5] = new Point3f(0, -330, -65);



        Point2f[] imagePoints = new Point2f[6];

        // Head Pose 추적 대표 점 Landmark (2D 이미지 좌표계)
        //NoseTop
        imagePoints[0] = (new Point2f(landmarks.Landmark[1].X * width, height - landmarks.Landmark[1].Y * height));
        //LeftEye
        imagePoints[1] = (new Point2f(landmarks.Landmark[33].X * width, height - landmarks.Landmark[33].Y * height));
        //RightEye
        imagePoints[2] = (new Point2f(landmarks.Landmark[263].X * width, height - landmarks.Landmark[263].Y * height));
        //MouthLeft
        imagePoints[3] = (new Point2f(landmarks.Landmark[61].X * width, height - landmarks.Landmark[61].Y * height));
        //MouthRight
        imagePoints[4] = (new Point2f(landmarks.Landmark[291].X * width, height - landmarks.Landmark[291].Y * height));
        //Chin
        imagePoints[5] = (new Point2f(landmarks.Landmark[199].X * width, height - landmarks.Landmark[199].Y * height));

        // 3D 좌표 행렬로 변환
        var object_points = new MatOfPoint3f(1, 6, objectPoints);
        // 2D 좌표 행렬로 변환
        var image_points = new MatOfPoint2f(1, 6, imagePoints);

        // 카메라 파라미터 행렬
        var camera_matrix = new Mat(3, 3, MatType.CV_32F, new float[] {   640, 0,   height / 2, 
                                                                            0,   640, width  / 2, 
                                                                            0,   0,   1          });

        // 왜곡보정 행렬
        var dist_coeffs = new Mat(1, 4, MatType.CV_32F, new float[] { 0, 0, 0, 0 });
        Mat rvec = new Mat(1, 3, MatType.CV_32F);
        Mat tvec =new Mat(1,3, MatType.CV_32F);

        // 3D 좌표계와 2D 좌표계, 카메라 파라미터, 왜곡 정보를 통해 카메라의 회전, 이동을 계산하는 함수
        Cv2.SolvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

        Mat rotMat = new Mat();

        // SolvePnP에서 나온 Rodrigues 표현식 rvec를 3x3 행렬로 변환
        Cv2.Rodrigues(rvec, rotMat);

        Mat mtxR = new Mat();
        Mat mtxQ = new Mat();


        // 3x3 행렬 정보를 변환하여 x,y,z 축에 대한 회전각도 출력
        Vec3d rotVec = Cv2.RQDecomp3x3(rotMat, mtxR, mtxQ);


        // OpenCV와 Unity 상의 좌표계 방향이 다르므로 부호를 바꾸어서 유니티 내부 Vector3 형식으로 저장
        Vector3 rotVecForUnity = new Vector3(-(float)rotVec.Item0, -(float)rotVec.Item1, (float)rotVec.Item2);


        // X축 이 상하, Y축이 좌우 패러미터
        cube.transform.localRotation = Quaternion.Euler((rotVecForUnity.x - TemporarycalibratedDirection.x), 
                                                        (rotVecForUnity.y - TemporarycalibratedDirection.y), 
                                                         rotVecForUnity.z - TemporarycalibratedDirection.z);


        //Debug.Log(rotVecForUnity);
        vec = rotVecForUnity;

        // normalize Head Pose
        Vector2 headPoseDir = new Vector2(rotVecForUnity.y/180, rotVecForUnity.x/180);

        return headPoseDir;
    }


    public void MoveCursorByGaze(float wx1, float wx2, float bx, float wy1, float wy2, float by)
    {
        Vector2 hp = HeadPoseTracker(GetComponent<FaceLandmarksParser>().FaceAndIrisLandmarks);
        Vector2 eye = EyeTracker(GetComponent<FaceLandmarksParser>().FaceAndIrisLandmarks);

        Vector2 newPos = new Vector2(((hp.x) * wx1 + (eye.x) * wx2 * 0.3f + bx) * UnityEngine.Screen.width/2,
                                     (-(hp.y) * wy1 + (eye.y) * wy2 * 0.3f + by) * UnityEngine.Screen.height/2);

        newPos.x = Mathf.Clamp(newPos.x, -width/2, width/2);
        newPos.y = Mathf.Clamp(newPos.y, -height/2, height/2);

        cursor.transform.localPosition = Vector2.Lerp(cursor.transform.localPosition, smoothing.DoSmoothing(newPos), Time.deltaTime * 20);
    }

    public void SaveCalibrationResult(float wx1, float wx2, float bx,
                                      float wy1, float wy2, float by)
    {
        XWeight1 = wx1;
        XWeight2 = wx2;
        YWeight1 = wy2;
        YWeight2 = wy2;
        XBias = bx;
        YBias = by;

        cursor.SetActive(true);
        isCalibreated = true;
    }


    //landmark의 정확한 위치 : x * width, -y*height, z * (깊이)
    public void FollowSphereToLandmarks(NormalizedLandmarkList landmarks)
    {
        for(int i = 0; i < 468; i++)
        {
            spheresForLandmarks[i].transform.localPosition = new Vector3(landmarks.Landmark[i].X* width - width/2, height/2-landmarks.Landmark[i].Y * height, landmarks.Landmark[i].Z * width - 50);
        }
    }



}

public class Smoothing
{
    private Vector2[] buffer;
    private int buffer_index;
    private Vector2 sum;

    public Smoothing(int num)
    {
        buffer = new Vector2[num];
        buffer_index = 0;
        sum = Vector2.zero;
    }
    public Vector2 DoSmoothing(Vector2 v)
    {
        sum -= buffer[buffer_index];
        buffer[buffer_index] = v;
        sum += v;

        buffer_index = (buffer_index + 1) % buffer.Length;

        return sum / buffer.Length;
    }
}