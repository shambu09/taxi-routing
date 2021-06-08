# taxi-routing

## Taxi routing based on passenger's destination.

-   Based on the number of passengers and their destinations, create their location groups [lat, long] using k-means clustering.
-   Find the distances of the location groups and the current location of the taxi and consider the nearest passenger.
-   Consider the passenger with minimum distance to the destination.
-   Find the shortest path to destination of a passenger whose destination is near to other location groups or other passengers for better profits.
