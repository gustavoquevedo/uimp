using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace EvalDataStreams1
{
    class Program
    {
        private const string Filename = "numbers.txt";
        private static Random rnd1 = new Random();
        private static Random rnd2 = new Random();
        private static Stopwatch sw = new Stopwatch();


        static void Main(string[] args)
        {
            var ejercicio1 = new Ejercicio1();
            ejercicio1.Run();

            //var ejercicio2 = new Ejercicio2();
            //ejercicio2.Run();

        }
    }
}
