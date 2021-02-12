using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EvalDataStreams1
{
    public static class GFG
    {
        // This code is contributed by Sam007_ 
        public static int GetMissingNo1(int[] a, int n)
        {
            int total = (n + 1) * (n + 2) / 2;

            for (int i = 0; i < n; i++)
                total -= a[i];

            return total;
        }

        // This code is contributed by SoumikMondal
        public static int GetMissingNo2(int[] a, int n)
        {
            int i, total = 1;

            for (i = 2; i <= (n + 1); i++)
            {
                total += i;
                total -= a[i - 2];
            }
            return total;
        }

        // This code is contributed by Sam007_
        public static int GetMissingNo3(int[] a, int n)
        {
            int x1 = a[0];
            int x2 = 1;

            /* For xor of all the elements 
            in array */
            for (int i = 1; i < n; i++)
                x1 = x1 ^ a[i];

            /* For xor of all the elements 
            from 1 to n+1 */
            for (int i = 2; i <= n + 1; i++)
                x2 = x2 ^ i;

            return (x1 ^ x2);
        }

        // This code is contributed by Virusbuddah
        public static int GetMissingNo4(int[] a,
                                int n)
        {
            int n_elements_sum = (n * (n + 1) / 2);
            int sum = 0;

            for (int i = 0; i < n - 1; i++)
                sum = sum + a[i];

            return (n_elements_sum - sum);
        }
    }


}
