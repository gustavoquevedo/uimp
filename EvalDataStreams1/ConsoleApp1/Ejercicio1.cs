using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace EvalDataStreams1
{
    public class Ejercicio1
    {
        private const string Filename = "numbers.txt";
        private static Random rnd2 = new Random();
        private static Stopwatch sw = new Stopwatch();

        private static int N = 10;

        public void Run()
        {
            int[] nValues = { 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000 };
            foreach (var n in nValues)
            {
                N = n;
                var sequence = GenerateSequence();

                sw.Restart();

                //Lectura/Escritura disco: sí                 
                WriteRandomSequenceFile(sequence);
                var missingNumber = GetMissingNumberFromFile();

                sw.Stop();
                Console.WriteLine("N={0}, Número={1}, Tiempo={2}, Lectura/Escritura disco: {3}", N, missingNumber, sw.Elapsed, "Sí");
            }
        }

        public static int GetMissingNumberFromFile()
        {
            //Leemos el fichero
            var sr = new StreamReader(Filename);
            //Leemos la primera línea con el primer número del fichero
            var line = sr.ReadLine();

            //Inicializamos las variables
            var total = 1;
            var index = 1;

            //Leemos hasta el final del fichero
            while (line != null)
            {
                //Incrementamos el índice y se lo sumamos a total
                total += (++index);
                //Restamos el último número leído de total
                total -= int.Parse(line);
                //Leemos un nuevo númreo del fichero
                line = sr.ReadLine();
            }
            //Cerrar fichero
            sr.Close();
            return total;
        }

        private static IEnumerable<int> GenerateSequence()
        {
            var missingNumber = rnd2.Next(N);
            missingNumber++;
            var intArray = new int[N - 1];
            var counter = 0;
            for (int i = 1; i < (N + 1); i++)
            {
                if (missingNumber != i)
                {
                    intArray[counter++] = i;
                }
            }
            return Common.Shuffle(intArray.ToList());
        }


        private static void WriteRandomSequenceFile(IEnumerable<int> sequence)
        {
            try
            {
                var sw = new StreamWriter(Filename);

                foreach (var i in sequence)
                {
                    sw.WriteLine(i);
                }
                sw.Close();
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
        }
    }
}
