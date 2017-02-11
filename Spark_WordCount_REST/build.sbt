import sbt.Keys._

name := "Spark_WordCount_REST"

version := "1.0"

scalaVersion := "2.11.8"

exportJars := true
fork := true

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.6.1",
  "net.databinder" %% "unfiltered-filter" % "0.8.3",
  "net.databinder" %% "unfiltered-jetty" % "0.8.3",
  "net.databinder" %% "unfiltered-directives" % "0.8.3"
)