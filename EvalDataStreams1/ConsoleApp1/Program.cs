using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
        private const string Filename = "numbers.txt";
        private static Random rnd1 = new Random();
        private static Random rnd2 = new Random();

        private static int N = 1000;

        static void Main(string[] args)
        {
            var sequence = GenerateSequence();
            WriteRandomSequenceFile(sequence);

            FindMissingNumber();
        }

        private static void FindMissingNumber()
        {
            String line;
            try
            {
                //Pass the file path and file name to the StreamReader constructor
                var sr = new StreamReader(Filename);
                //Read the first line of text
                line = sr.ReadLine();
                //Continue to read until you reach end of file
                while (line != null)
                {
                    //write the lie to console window
                    Console.WriteLine(line);
                    //Read the next line
                    line = sr.ReadLine();
                }
                //close the file
                sr.Close();
                Console.ReadLine();
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

        private static IEnumerable<int> GenerateSequence()
        {
            var missingNumber = rnd2.Next(N);
            var intArray = new int[N - 1];
            var counter = 0;
            for (int i = 0; i < N; i++)
            {
                if (missingNumber != i)
                {
                    intArray[counter++] = i;
                }
            }
            return Shuffle(intArray.ToList());
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
    }
}
