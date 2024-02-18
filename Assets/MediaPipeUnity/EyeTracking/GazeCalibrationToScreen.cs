using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GazeCalibrationToScreen : MonoBehaviour
{
    [SerializeField] private float width;
    [SerializeField] private float height;


    // 눈동자(왼쪽 혹은 오른쪽 중 한 가지만 참조)
    private float eyeX, eyeY;
    // 머리 방향(가로, 세로) 정보
    private float headPoseHorizontal, headPoseVertical; // -> -180~180 사이의 값이므로 Normalize 필요

    /*
     보고있는 좌표로 위치를 예측하는 함수
     예측값 = ((headPoseHorizontal/180+eyeX) * a + b, (headPoseVertical/180+eyeY) * c + d) 
     각 X 축과 Y 축에 대한 선형 회귀 분석 진행
     */

    private Vector2[] Coordinates;

    // Start is called before the first frame update
    void Start()
    {
        // 캘리브레이션에 필요한 임의의 점 9개
        Coordinates = new Vector2[] {   new Vector2(100,100),
                                        new Vector2(300,100),
                                        new Vector2(500,100),
                                        new Vector2(100,250),
                                        new Vector2(300,250),
                                        new Vector2(500,250),
                                        new Vector2(100,400),
                                        new Vector2(300,400),
                                        new Vector2(500,400),};
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
