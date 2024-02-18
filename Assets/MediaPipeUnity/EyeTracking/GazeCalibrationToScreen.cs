using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GazeCalibrationToScreen : MonoBehaviour
{
    [SerializeField] private float width;
    [SerializeField] private float height;


    // ������(���� Ȥ�� ������ �� �� ������ ����)
    private float eyeX, eyeY;
    // �Ӹ� ����(����, ����) ����
    private float headPoseHorizontal, headPoseVertical; // -> -180~180 ������ ���̹Ƿ� Normalize �ʿ�

    /*
     �����ִ� ��ǥ�� ��ġ�� �����ϴ� �Լ�
     ������ = ((headPoseHorizontal/180+eyeX) * a + b, (headPoseVertical/180+eyeY) * c + d) 
     �� X ��� Y �࿡ ���� ���� ȸ�� �м� ����
     */

    private Vector2[] Coordinates;

    // Start is called before the first frame update
    void Start()
    {
        // Ķ���극�̼ǿ� �ʿ��� ������ �� 9��
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
