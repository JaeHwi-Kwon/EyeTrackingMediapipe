using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Mediapipe;
using Mediapipe.Unity;
using Mediapipe.Unity.CoordinateSystem;
using UnityEngine.UI;
using System.Linq;

public class GazeEstimator : MonoBehaviour
{
    [SerializeField] private RawImage screen;

    private UnityEngine.Rect screenRect;
    private NormalizedLandmarkList faceNirislandmarks;

    // Start is called before the first frame update
    void Start()
    {
        screenRect = screen.GetComponent<RectTransform>().rect;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void HeadPoseLogger(NormalizedLandmarkList landmarks)
    {
        Vector2[] face_2d = new Vector2[6];
        Vector3[] face_3d = new Vector3[6];

        Vector3 nosetop = screenRect.GetPoint(landmarks.Landmark[1]);
        Vector3 lefteye = screenRect.GetPoint(landmarks.Landmark[33]);
        Vector3 righteye = screenRect.GetPoint(landmarks.Landmark[263]);
        Vector3 mouthleft = screenRect.GetPoint(landmarks.Landmark[61]);
        Vector3 mouthright = screenRect.GetPoint(landmarks.Landmark[291]);
        Vector3 chin = screenRect.GetPoint(landmarks.Landmark[199]);

        face_3d.Append(nosetop);
        face_2d.Append(new Vector2(nosetop.x, nosetop.y));
        face_3d.Append(lefteye);
        face_2d.Append(new Vector2(lefteye.x, lefteye.y));
        face_3d.Append(righteye);
        face_2d.Append(new Vector2(righteye.x, righteye.y));
        face_3d.Append(mouthleft);
        face_2d.Append(new Vector2(mouthleft.x, mouthleft.y));
        face_3d.Append(mouthright);
        face_2d.Append(new Vector2(mouthright.x, mouthright.y));
        face_3d.Append(chin);
        face_2d.Append(new Vector2(chin.x, chin.y));


    }
}
