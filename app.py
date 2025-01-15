from flask import Flask, request, send_file
from engraver import generate_aruco_pattern, combine_with_base

app = Flask(__name__)

@app.route('/')
def index():
    return '''
        <form action="/generate" method="post">
            <label for="number">Enter a number:</label>
            <input type="number" id="number" name="number">
            <button type="submit">Generate STL</button>
        </form>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    number = int(request.form['number'])
    # Generate the ArUco pattern
    pattern = generate_aruco_pattern(number)
    # Generate the STL file
    output_file = "Printing_Tag.stl"
    combine_with_base(
        array=pattern,
        pattern_position=(-30, -30, 102),
        rotation=(0, 0, 0),
        output_file=output_file
    )
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
