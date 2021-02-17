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

        private static int N = 10;

        static void Main(string[] args)
        {
            int[] nValues = { 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000 };
            foreach(var n in nValues)
            {
                N = n;
                var sequence = GenerateSequence();

                //WriteRandomSequenceFile(sequence);

                //var numbers = ReadFile();
                var numbers = sequence.ToArray();

                //GetMissingNumber(numbers, GFG.GetMissingNo1);
                GetMissingNumber(numbers, GFG.GetMissingNo2);
                GetMissingNumber(numbers, GFG.GetMissingNo3);
            }
        }

        private static void GetMissingNumber(int[] numbers, Func<int[], int, int> getMissingNumber)
        {
            sw.Restart();
            var missingNumber = getMissingNumber(numbers, N - 1);
            sw.Stop();
            Console.WriteLine("N={0}, Número={1}, Tiempo={2}, Función={3}", N, missingNumber, sw.Elapsed, getMissingNumber.Method.Name);
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
            return Shuffle(intArray.ToList());
        }

        public static IEnumerable<T> Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rnd1.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
            return list;
        }

        private static void WriteRandomSequenceFile(IEnumerable<int> sequence)
        {
            try
            {
                //Pass the filepath and filename to the StreamWriter Constructor
                var sw = new StreamWriter(Filename);

                foreach (var i in sequence)
                {
                    sw.WriteLine(i);
                }
                //Close the file
                sw.Close();
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
            finally
            {
                Console.WriteLine("Executing finally block.");
            }
        }

        private static int[] ReadFile()
        {
            var numbers = new List<int>();
            //Pass the file path and file name to the StreamReader constructor
            var sr = new StreamReader(Filename);
            //Read the first line of text
            var line = sr.ReadLine();
            //Continue to read until you reach end of file
            while (line != null)
            {
                numbers.Add(int.Parse(line));
                line = sr.ReadLine();
            }
            //close the file
            sr.Close();
            return numbers.ToArray();
        }
    }
}
