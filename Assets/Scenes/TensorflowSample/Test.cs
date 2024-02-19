using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using System.IO;
using UnityEngine.UIElements;
using System;
using Tensorflow.Debugging;

public class Test : MonoBehaviour
{

    private Vector2[] Coordinates;
    private Vector2[] HeadPose;
    private Vector2[] EyeDir;
    public int training_epochs = 1000;
    private float learning_rate = 0.01f;
    private int display_step = 50;

    private NDArray train_X, train_Y;
    int n_samples;

    private ICallback result;
    // Start is called before the first frame update
    void Start()
    {

        Coordinates = new Vector2[] {   new Vector2(100,100),
                                        new Vector2(300,100),
                                        new Vector2(500,100),
                                        new Vector2(100,250),
                                        new Vector2(300,250),
                                        new Vector2(500,250),
                                        new Vector2(100,400),
                                        new Vector2(300,400),
                                        new Vector2(500,400)};

        HeadPose = new Vector2[] {   new Vector2(0,0),
                                        new Vector2(0,0),
                                        new Vector2(0,0),
                                        new Vector2(0,0),
                                        new Vector2(0,0),
                                        new Vector2(0,0),
                                        new Vector2(0,0),
                                        new Vector2(0,0),
                                        new Vector2(0,0)};


        EyeDir = new Vector2[] {   new Vector2(-0.8f,-0.8f),
                                        new Vector2(0,-0.8f),
                                        new Vector2(0.8f,-0.8f),
                                        new Vector2(-0.8f,0),
                                        new Vector2(0,0),
                                        new Vector2(0.8f,0),
                                        new Vector2(-0.8f,0.8f),
                                        new Vector2(0,0.8f),
                                        new Vector2(0.8f,0.8f)};

        bool success = RunLinearRegression();

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private bool RunLinearRegression()
    {
        tf.compat.v1.disable_eager_execution();

        PrepareData(EyeDir,HeadPose);

        var X = tf.placeholder(tf.float32);
        var Y= tf.placeholder(tf.float32);

        var W = tf.Variable(-0.06f, name: "weight");
        var b = tf.Variable(-0.73f, name: "bias");

        var pred = tf.add(tf.multiply(X, W), b);

        var cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * n_samples);

        var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

        var init = tf.global_variables_initializer();

        using var sess = tf.Session();

        sess.run(init);

        // Fit all training data
        for (int epoch = 0; epoch < training_epochs; epoch++)
        {
            foreach (var (x, y) in zip<float>(train_X, train_Y))
                sess.run(optimizer, (X, x), (Y, y));

            // Display logs per epoch step
            if ((epoch + 1) % display_step == 0)
            {
                var c = sess.run(cost, (X, train_X), (Y, train_Y));
                Debug.Log($"Epoch: {epoch + 1} cost={c} " + $"W={sess.run(W)} b={sess.run(b)}");
            }
        }

        Console.WriteLine("Optimization Finished!");
        var training_cost = sess.run(cost, (X, train_X), (Y, train_Y));
        Console.WriteLine($"Training cost={training_cost} W={sess.run(W)} b={sess.run(b)}");

        // Testing example
        var test_X = np.array(6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f);
        var test_Y = np.array(1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f);
        Console.WriteLine("Testing... (Mean square loss Comparison)");
        var testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * test_X.shape[0]),
            (X, test_X), (Y, test_Y));
        Console.WriteLine($"Testing cost={testing_cost}");
        var diff = Math.Abs((float)training_cost - (float)testing_cost);
        Console.WriteLine($"Absolute mean square loss difference: {diff}");

        return diff < 0.01;
    }


    private void PrepareData(Vector2[] eye, Vector2[] head)
    {
        float[] GazeX = new float[9];
        float[] coordinatesX = new float[9];
        for (int i = 0; i < 9; i++)
        {
            GazeX[i] = eye[i].x + head[i].x;
            coordinatesX[i] = Coordinates[i].x;
        }


        train_X = np.array(GazeX);
        train_Y = np.array(coordinatesX);

        n_samples = (int)train_X.shape[0];
    }
}
